"""
预审 Agent 调度框架（preaudit_agent）

从固定流水线升级为 Agent 架构：
- Planner：根据申请情况规划需要调用的工具
- Tools：OCR抽取 / 分层规则 / Hybrid RAG / 知识图谱 / 外部核验 / LLM解释
- Executor：动态调用工具，结果回流
- Reporter：生成结构化审核报告

核心区别（vs 旧流水线）：
1. 动态规划 — 不是每次都跑全部步骤，根据材料情况决定调哪些工具
2. 风险驱动检索 — 发现异常后主动生成问题去检索 RAG（而非被动检索）
3. LLM 重新定位 — 只做理解+解释，不做通过/拒绝决策
4. 结构化输出 — 规则发现 + 风险评分 + 审核建议，可追溯可解释
"""
import os
import json
import re
import time
from typing import Optional

import config
from api_clients import get_qwen
from business_entry import detect_product, load_strategy
from layered_rules import run_all_layers
from risk_scoring import calculate_risk_score
import review_loop


# ============================================================
# 工具集
# ============================================================
class PreauditTools:
    """Agent 可调用的工具集合"""

    @staticmethod
    def tool_ocr_extract(file_paths: list, progress_cb=None) -> dict:
        """工具1: OCR + 分类 + 抽取"""
        import step2_classify_extract
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if progress_cb: progress_cb(10, "OCR + 分类 + 抽取")
        docs = [None] * len(file_paths)
        def _w(i, f):
            try: return i, step2_classify_extract.classify_and_extract(f)
            except Exception as e:
                return i, {"doc_type": "unknown", "confidence": 0, "fields": {},
                           "ocr_text": "", "notes": str(e), "file": f}
        with ThreadPoolExecutor(max_workers=min(4, len(file_paths))) as pool:
            futs = [pool.submit(_w, i, f) for i, f in enumerate(file_paths)]
            for fut in as_completed(futs):
                idx, r = fut.result()
                docs[idx] = r
        valid = [d for d in docs if d.get("quality_flag") not in ("ocr_failed", "irrelevant", "ocr_empty")]
        return {"extracted_docs": valid, "rejected": len(docs) - len(valid), "all_docs": docs}

    @staticmethod
    def tool_business_entry(extracted_docs: list, application_text: str = "") -> dict:
        """工具2: 产品识别 + 策略加载"""
        return detect_product(extracted_docs, application_text)

    @staticmethod
    def tool_layered_rules(extracted_docs: list, strategy: dict,
                           rule_result: dict = None, kg_alerts: list = None) -> dict:
        """工具3: 四层规则引擎"""
        return run_all_layers(extracted_docs, strategy, rule_result, kg_alerts)

    @staticmethod
    def tool_rag_search(risk_findings: list, strategy: dict, progress_cb=None) -> dict:
        """工具4: 风险驱动 Hybrid RAG 检索
        发现异常后主动生成问题去检索对应规则
        """
        if progress_cb: progress_cb(72, "风险驱动 RAG 检索")
        try:
            import step10_rag_decision
            # 风险驱动：根据 findings 的 category 生成检索查询
            risk_queries = _generate_risk_queries(risk_findings, strategy)
            result = step10_rag_decision.enhance_decision_with_rag(
                risk_points=[{"type": f["rule_name"], "detail": f["evidence"]}
                             for f in risk_findings if f["severity"] in ("block", "major")],
                rule_issues=[{"level": "warn" if f["severity"] == "minor" else "reject",
                              "msg": f["evidence"]} for f in risk_findings],
                cross_doc_issues=[],
                missing_result={"required_missing": []},
                loan_product=strategy.get("name", ""),
            )
            result["risk_queries"] = risk_queries
            return result
        except Exception as e:
            return {"error": str(e), "regulation_references": [], "policy_references": [],
                    "case_references": [], "structured_matches": [], "risk_queries": []}

    @staticmethod
    def tool_kg_fraud(person: str, extracted_docs: list, progress_cb=None,
                      all_applicants_data: list = None) -> dict:
        """工具5: 知识图谱反欺诈
        :param all_applicants_data: 批量模式下所有申请人的抽取数据，用于跨申请人团伙检测
        """
        if progress_cb: progress_cb(82, "知识图谱反欺诈")
        try:
            import step11_kg_fraud
            return step11_kg_fraud.enhance_decision_with_kg(
                person=person, extracted_docs=extracted_docs,
                all_applicants_data=all_applicants_data,
            )
        except Exception as e:
            return {"kg_alerts": [], "has_fraud_network": False, "error": str(e)}

    @staticmethod
    def tool_risk_scoring(layered_result: dict, extracted_docs: list, strategy: dict,
                          rag_result: dict, kg_result: dict, inference_result: dict) -> dict:
        """工具6: 多源风险评分"""
        return calculate_risk_score(
            layered_result, extracted_docs, strategy,
            rag_result=rag_result,
            kg_alerts=kg_result.get("kg_alerts", []) if kg_result else None,
            inference_result=inference_result,
        )

    @staticmethod
    def tool_llm_explain(extracted_docs: list, layered_result: dict, risk_score: dict,
                         strategy: dict, progress_cb=None) -> dict:
        """工具7: LLM 风险解释（不决策，只解释）
        LLM 定位：理解复杂情况 + 生成审核意见，不决定通过/拒绝
        """
        if progress_cb: progress_cb(92, "LLM 风险解释生成")
        try:
            import step7_inference_risk
            issues = [{"level": "reject" if f["severity"] == "block" else "warn",
                       "msg": f["evidence"]} for f in layered_result.get("findings", [])]
            return step7_inference_risk.assess_semantic_risk(
                extracted_docs, rule_issues=issues, cross_doc_issues=[],
            )
        except Exception as e:
            return {"risk_level": risk_score.get("risk_level", "unknown"),
                    "risk_points": [], "error": str(e)}


def _generate_risk_queries(findings: list, strategy: dict) -> list:
    """风险驱动：根据发现的问题 + 策略触发词生成 RAG 检索查询"""
    queries = []
    triggers = strategy.get("risk_search_triggers", {})
    # 规则 category → 触发词映射
    category_map = {
        "income_mismatch": "income_mismatch",
        "overdue_excess": "overdue_excess",
        "id_mismatch": "id_mismatch",
        "high_dti": "high_dti",
        "license_expired": "license_expired",
        "business_age_short": "business_age_short",
        "property_dispute": "property_dispute",
        "marriage_status": "marriage_status",
    }
    triggered_keys = set()
    for f in findings:
        # 根据规则名/证据推断触发词
        name = f.get("rule_name", "") + f.get("evidence", "")
        if "收入" in name or "income" in name.lower():
            triggered_keys.add("income_mismatch")
        if "逾期" in name or "overdue" in name.lower():
            triggered_keys.add("overdue_excess")
        if "身份证" in name and "不一致" in name:
            triggered_keys.add("id_mismatch")
        if "DTI" in name or "负债比" in name:
            triggered_keys.add("high_dti")
        if "资质" in name or "过期" in name or "license" in name.lower():
            triggered_keys.add("license_expired")
        if "房产" in name or "不动产" in name or "产权" in name:
            triggered_keys.add("property_dispute")
        if "婚姻" in name or "配偶" in name:
            triggered_keys.add("marriage_status")

    for key in triggered_keys:
        kws = triggers.get(key, [])
        if kws:
            queries.append({"trigger": key, "keywords": kws, "query": " ".join(kws)})

    return queries


# ============================================================
# Agent 规划器
# ============================================================
def plan_tasks(extracted_docs: list, strategy: dict) -> list:
    """
    根据材料情况规划需要执行的任务
    不是每次都跑全部步骤，按需调用
    """
    plan = ["business_entry", "layered_rules"]
    doc_types = [d.get("doc_type", "") for d in extracted_docs]

    # 有风险材料 → 触发 RAG 检索
    has_risk_doc = any(t in doc_types for t in ("credit_report", "bank_statement", "business_license"))
    if has_risk_doc:
        plan.append("rag_search")

    # 始终跑 KG 反欺诈（轻量）
    plan.append("kg_fraud")

    # 始终跑风险评分
    plan.append("risk_scoring")

    # 中高风险 → LLM 解释
    plan.append("llm_explain")

    return plan


# ============================================================
# Agent 主循环
# ============================================================
def run_preaudit_agent(person: str, file_paths: list,
                       loan_product_hint: str = "",
                       progress_callback=None,
                       all_applicants_data: list = None) -> dict:
    """
    预审 Agent 主入口

    :param person: 申请人
    :param file_paths: 材料文件路径列表
    :param loan_product_hint: 产品提示（可空，Agent 自动识别）
    :param progress_callback: fn(progress, step) 进度回调
    :param all_applicants_data: 批量模式下所有申请人的抽取数据，用于 KG 跨申请人团伙检测
    :return: 结构化审核报告
    """
    t0 = time.time()
    def _prog(p, s, extra=None):
        if progress_callback:
            try: progress_callback(p, s, extra)
            except Exception: pass

    print(f"\n{'='*60}")
    print(f"  预审 Agent 启动: {person} ({len(file_paths)} 份材料)")
    print(f"{'='*60}")

    # ---- Step 0: OCR + 抽取 ----
    _prog(5, "OCR + 分类 + 抽取")
    ocr_result = PreauditTools.tool_ocr_extract(file_paths, _prog)
    extracted_docs = ocr_result["extracted_docs"]
    if not extracted_docs:
        return _build_error_report(person, "all_ocr_failed", file_paths, t0)
    print(f"  [Agent] OCR 完成: {len(extracted_docs)} 份有效材料")

    # ---- Step 1: 业务入口（产品识别）----
    _prog(20, "产品识别 + 策略加载")
    app_text = ""
    for d in extracted_docs:
        if d.get("doc_type") == "loan_application":
            app_text = d.get("ocr_text", "")
            break
    if loan_product_hint and loan_product_hint in __import__("business_entry").PRODUCT_STRATEGIES:
        strategy = load_strategy(loan_product_hint)
        product_info = {"product_key": loan_product_hint, "product_name": strategy["name"],
                        "confidence": 1.0, "evidence": "外部指定", "strategy": strategy}
    else:
        product_info = PreauditTools.tool_business_entry(extracted_docs, app_text)
        strategy = product_info["strategy"]
    print(f"  [Agent] 产品识别: {product_info['product_name']} (conf={product_info['confidence']})")
    _prog(22, "产品识别完成", {"product_info": {
        "product_key": product_info["product_key"],
        "product_name": product_info["product_name"],
        "confidence": product_info["confidence"],
        "evidence": product_info["evidence"],
    }})

    # ---- Step 2: 规划任务 ----
    plan = plan_tasks(extracted_docs, strategy)
    print(f"  [Agent] 任务规划: {' → '.join(plan)}")

    # ---- Step 3: 分层规则引擎 ----
    _prog(35, "四层规则引擎")
    # 先跑 step6 规则引擎拿流水月均（兼容）
    rule_result = {"avg_monthly_income": _extract_avg_income(extracted_docs)}
    layered_result = PreauditTools.tool_layered_rules(
        extracted_docs, strategy, rule_result=rule_result)
    print(f"  [Agent] 规则引擎: {len(layered_result['findings'])} 条发现 "
          f"(block={layered_result['block_count']}, major={layered_result['major_count']})")

    # ---- Step 4: KG 反欺诈 ----
    kg_result = None
    if "kg_fraud" in plan:
        kg_result = PreauditTools.tool_kg_fraud(person, extracted_docs, _prog,
                                                 all_applicants_data=all_applicants_data)
        kg_alerts = kg_result.get("kg_alerts", [])
        if kg_alerts:
            print(f"  [Agent] KG 告警: {len(kg_alerts)} 条")
            # KG 告警补充到规则引擎 L4
            layered_result = PreauditTools.tool_layered_rules(
                extracted_docs, strategy, rule_result=rule_result, kg_alerts=kg_alerts)

    # ---- Step 5: 风险驱动 RAG 检索 ----
    rag_result = None
    if "rag_search" in plan:
        rag_result = PreauditTools.tool_rag_search(
            layered_result.get("findings", []), strategy, _prog)
        rq = rag_result.get("risk_queries", [])
        if rq:
            print(f"  [Agent] 风险驱动检索: {len(rq)} 个触发 → "
                  f"{len(rag_result.get('regulation_references', []))} 法规 + "
                  f"{len(rag_result.get('policy_references', []))} 制度")

    # ---- Step 6: LLM 风险解释 ----
    inference_result = None
    if "llm_explain" in plan:
        # 先算初步评分给 LLM 参考
        prelim_score = calculate_risk_score(
            layered_result, extracted_docs, strategy,
            rag_result=rag_result,
            kg_alerts=kg_result.get("kg_alerts", []) if kg_result else None,
            inference_result=None,
        )
        inference_result = PreauditTools.tool_llm_explain(
            extracted_docs, layered_result, prelim_score, strategy, _prog)
        print(f"  [Agent] LLM 解释: {inference_result.get('risk_level', '?')} "
              f"({len(inference_result.get('risk_points', []))} 个风险点)")

    # ---- Step 7: 综合风险评分 ----
    _prog(95, "综合风险评分")
    risk_score = PreauditTools.tool_risk_scoring(
        layered_result, extracted_docs, strategy,
        rag_result or {}, kg_result or {}, inference_result or {})
    print(f"  [Agent] 风险评分: {risk_score['total_score']} → {risk_score['action_label']}")

    # ---- Step 8: 生成报告 ----
    _prog(98, "生成审核报告")
    report = _build_report(
        person, extracted_docs, product_info, strategy, layered_result,
        risk_score, rag_result, kg_result, inference_result, ocr_result, t0)

    _prog(100, "完成")
    print(f"\n  决策: {report['action']} | 风险分: {risk_score['total_score']} | 耗时: {time.time()-t0:.1f}s")
    return report


# ============================================================
# 辅助函数
# ============================================================
def _extract_avg_income(extracted_docs: list) -> float:
    """从流水估算月均收入
    兼容多种字段命名：
    - schema 标准：income/expense 分离字段
    - 带符号 amount：正数=收入，负数=支出
    - LLM 偶发误填：name 字段存了户名
    """
    for d in extracted_docs:
        if d.get("doc_type") == "bank_statement":
            txns = d.get("transactions") or []
            if not txns:
                continue
            income_sum = 0.0
            for t in txns:
                try:
                    # 优先读 income 字段（schema 标准）
                    if "income" in t and t.get("income") is not None:
                        amt = float(t.get("income") or 0)
                        if amt > 0:
                            income_sum += amt
                            continue
                    # 兼容 amount 字段（正数=收入）
                    if "amount" in t and t.get("amount") is not None:
                        s = str(t.get("amount")).strip()
                        s_clean = re.sub(r"[^\d.\-]", "", s)
                        amt = float(s_clean)
                        if amt > 0:
                            income_sum += amt
                except (ValueError, TypeError):
                    continue
            months = set()
            for t in txns:
                ds = t.get("date") or ""
                # 兼容多种日期格式：2025-01-05 / 20250105 / 2025/01/05
                # 统一归一化为 YYYY-MM，避免 "20250105"[:7]="2025010" 导致同月不同日被算成不同月份
                m = re.search(r"(\d{4})[-/]?(\d{2})", ds)
                if m:
                    months.add(f"{m.group(1)}-{m.group(2)}")
            if months:
                return income_sum / len(months)
    return 0


def _build_report(person, extracted_docs, product_info, strategy, layered_result,
                  risk_score, rag_result, kg_result, inference_result, ocr_result, t0):
    """生成结构化审核报告"""
    findings = layered_result.get("findings", [])
    # 补充 RAG policy_basis 到 findings
    if rag_result:
        reg_refs = rag_result.get("regulation_references", []) + rag_result.get("policy_references", [])
        for f in findings:
            if not f.get("policy_basis") and reg_refs:
                f["policy_basis"] = reg_refs[0].get("article", "")

    # 字段差异（从 L2 findings 提取）
    # values 从 extracted_docs 中按规则相关的字段名提取实际值
    _FIELD_KEYS_BY_RULE = {
        "L2-02": ["id_number", "身份证号", "公民身份号码"],
        "L2-03": ["name", "姓名"],
        "L2-06": ["account_holder", "account_name", "户名", "company_name", "企业名称", "名称", "name"],
        "L2-07": ["loan_amount", "贷款金额", "申请金额", "loan_term", "贷款期限", "position", "职位", "annual_income", "年收入"],
    }
    discrepancies = []
    for f in findings:
        if f["layer"] == "L2" and f["severity"] in ("block", "major"):
            rule_id = f.get("rule_id", "")
            field_keys = _FIELD_KEYS_BY_RULE.get(rule_id, [])
            values = []
            for doc_type in f.get("docs_involved", []):
                # 在 extracted_docs 中找对应 doc_type 的材料
                actual_value = ""
                for d in extracted_docs:
                    if d.get("doc_type") == doc_type:
                        fld = d.get("fields", {})
                        for k in field_keys:
                            v = fld.get(k)
                            if v is not None and str(v).strip():
                                actual_value = str(v).strip()
                                break
                        break
                values.append({"doc": doc_type, "value": actual_value})
            discrepancies.append({
                "field": f["rule_name"],
                "judgment": "不一致" if "不一致" in f["rule_name"] or "冲突" in f["rule_name"] else "异常",
                "values": values,
                "evidence": f.get("evidence", ""),
            })

    # 缺失材料（从 L1 findings 提取）
    missing_required = [f["evidence"].replace("缺失必填材料：", "").split("（")[0]
                        for f in findings if f["rule_id"] == "L1-01"]
    missing_optional = [f["evidence"].replace("缺失选填材料：", "").split("（")[0]
                        for f in findings if f["rule_id"] == "L1-02"]

    # OCR 抽取结果摘要（供前端展示）
    ocr_results = []
    for d in extracted_docs:
        ocr_results.append({
            "doc_type": d.get("doc_type", "unknown"),
            "file_name": d.get("file_name", d.get("source", "")),
            "ocr_confidence": d.get("ocr_confidence", 0.85),
            "fields": d.get("fields", {}),
        })

    # 从 dimension_scores 提取置信度分量（统一为 0-1 范围）
    dim = risk_score.get("dimension_scores", {})
    def _dim_score(key, default):
        v = dim.get(key)
        if isinstance(v, dict):
            s = v.get("score", default)
        elif isinstance(v, (int, float)):
            s = v
        else:
            s = default
        return round(s / 100, 3) if s > 1 else round(s, 3)
    confidence = {
        "total_confidence": round(risk_score["total_score"] / 100, 3),
        "decision": risk_score["action"],
        "ocr_confidence": _dim_score("ocr_confidence", 85),
        "rule_confidence": _dim_score("business_rule", 80),
        "inference_confidence": _dim_score("llm_semantic", 75),
        "data_completeness": _dim_score("completeness", 90),
    }

    # reject_reason 兼容
    reject_reason = ""
    if risk_score["action"] == "reject":
        reject_reason = risk_score.get("explanation", "")
        if not reject_reason and risk_score.get("rule_block_override"):
            block_findings = [f for f in findings if f.get("severity") == "block"]
            if block_findings:
                reject_reason = "触发一票否决规则：" + "；".join(f.get("rule_name", "") for f in block_findings[:3])

    elapsed = time.time() - t0
    return {
        "person": person,
        "doc_count": len(extracted_docs),
        "doc_types": [d.get("doc_type") for d in extracted_docs],
        "ocr_results": ocr_results,
        "product": product_info,
        "strategy_applied": strategy["name"],
        # 规则发现
        "findings": findings,
        "layer_summary": layered_result["layer_summary"],
        "severity_summary": layered_result["severity_summary"],
        # 风险评分
        "risk_score": risk_score["total_score"],
        "risk_level": risk_score["risk_level"],
        "action": risk_score["action"],
        "action_label": risk_score["action_label"],
        "action_explanation": risk_score["explanation"],
        "dimension_scores": risk_score["dimension_scores"],
        "rule_block_override": risk_score["rule_block_override"],
        # 风险点（LLM 解释）
        "risk_points": (inference_result or {}).get("risk_points", []),
        # 字段差异 + 缺失
        "field_discrepancies": discrepancies,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "reject_reason": reject_reason,
        # RAG
        "rag_result": rag_result,
        # KG
        "kg_alerts": (kg_result or {}).get("kg_alerts", []),
        # 兼容旧接口
        "decision": risk_score["decision"],
        "confidence": confidence,
        "risk_points_count": len((inference_result or {}).get("risk_points", [])),
        "elapsed_seconds": round(elapsed, 1),
        # Agent 元信息
        "agent_version": "2.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _build_error_report(person, error, file_paths, t0):
    return {
        "person": person,
        "error": error,
        "doc_count": 0,
        "action": "reject",
        "action_label": "拒绝",
        "risk_score": 0,
        "risk_level": "high",
        "elapsed_seconds": round(time.time() - t0, 1),
        "agent_version": "2.0",
    }


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    # 用 12-谢衍 的真实材料测试（全量）
    from fcmbench_loader import IMAGES_DIR
    import os
    person_dir = "12-谢衍"
    p = os.path.join(IMAGES_DIR, person_dir)
    if not os.path.exists(p):
        print(f"测试目录不存在: {p}")
        exit()

    # 取全部单材料目录的"正常"图
    _COMBO_DIRS = {"组合1", "组合2", "组合3", "组合4", "组合6"}
    file_paths = []
    for sub in sorted(os.listdir(p)):
        sp = os.path.join(p, sub)
        if not os.path.isdir(sp) or sub in _COMBO_DIRS:
            continue
        for f in os.listdir(sp):
            if "正常" in f:
                file_paths.append(os.path.join(sp, f))
                break

    print(f"测试材料: {len(file_paths)} 份")
    for fp in file_paths:
        print(f"  - {os.path.basename(os.path.dirname(fp))}/{os.path.basename(fp)}")

    report = run_preaudit_agent(person_dir, file_paths, progress_callback=lambda p, s: print(f"  [{p}%] {s}"))

    print(f"\n{'='*60}")
    print(f"  审核报告")
    print(f"{'='*60}")
    print(f"申请人: {report.get('person')}")
    print(f"产品: {report.get('strategy_applied')}")
    print(f"风险分: {report.get('risk_score')}  等级: {report.get('risk_level')}")
    print(f"动作: {report.get('action_label')}  理由: {report.get('action_explanation')}")
    print(f"\n分层统计: {report.get('layer_summary')}")
    print(f"严重度: {report.get('severity_summary')}")
    print(f"\n维度评分:")
    for dim, d in (report.get('dimension_scores') or {}).items():
        print(f"  {dim:16s}: {d['score']:3d}  ({d['detail']})")
    print(f"\n规则发现 ({len(report.get('findings', []))} 条):")
    for f in report.get("findings", []):
        print(f"  [{f['layer']}] {f['rule_id']} {f['severity']:6s} {f['rule_name']}: {f['evidence'][:50]}")
    print(f"\n耗时: {report.get('elapsed_seconds')}s")
