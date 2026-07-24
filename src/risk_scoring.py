"""
风险评分层（risk_scoring）

将多源证据融合为 0-100 综合风险分数，输出三档分流：
- 低风险（≥80）：自动通过
- 中风险（50-79）：人工复核
- 高风险（<50）：重点调查 / 拒绝

多源证据融合（按产品策略权重加权）：
1. 规则风险分（L1-L4 四层规则，block/major/minor 加权扣分）
2. OCR 可信度分（抽取置信度均值）
3. 跨文档差异分（L2 层发现数转化）
4. RAG 证据分（是否有不利条款匹配）
5. 知识图谱风险分（KG 告警数转化）
6. LLM 语义风险分（推理风险点数 + 风险等级）

核心原则：
- 规则引擎做硬判断（block 一票否决）
- 评分模型做软融合（多源证据加权）
- LLM 只提供语义理解，不决定通过/拒绝
- 最终分流由综合分 + 规则 block 共同决定
"""
from typing import Optional


# ============================================================
# 各源评分（每源输出 0-100 分，100 = 最安全，0 = 最高风险）
# ============================================================
def score_rules(layered_result: dict) -> dict:
    """规则风险分：根据四层规则的 severity 扣分"""
    sev = layered_result.get("severity_summary", {})
    block_n = sev.get("block", 0)
    major_n = sev.get("major", 0)
    minor_n = sev.get("minor", 0)
    info_n = sev.get("info", 0)

    # 扣分：block=-25/条, major=-12/条, minor=-4/条, info=-1/条
    penalty = block_n * 25 + major_n * 12 + minor_n * 4 + info_n * 1
    score = max(0, 100 - penalty)

    return {
        "score": score,
        "detail": f"block×{block_n}(-{block_n*25}) + major×{major_n}(-{major_n*12}) + minor×{minor_n}(-{minor_n*4})",
        "block_count": block_n,
    }


def score_ocr_confidence(extracted_docs: list) -> dict:
    """OCR 可信度分：抽取置信度均值"""
    confs = [d.get("confidence", 0) for d in extracted_docs if d.get("confidence", 0) > 0]
    if not confs:
        return {"score": 50, "detail": "无置信度数据"}
    avg = sum(confs) / len(confs)
    return {"score": round(avg * 100), "detail": f"平均置信度 {avg:.3f}（{len(confs)} 份材料）"}


def score_cross_doc(layered_result: dict) -> dict:
    """跨文档差异分：L2 层发现数转化"""
    l2_findings = [f for f in layered_result.get("findings", []) if f["layer"] == "L2"]
    block_n = sum(1 for f in l2_findings if f["severity"] == "block")
    major_n = sum(1 for f in l2_findings if f["severity"] == "major")
    # 每个 block L2 扣 20，major 扣 10
    penalty = block_n * 20 + major_n * 10
    score = max(0, 100 - penalty)
    return {
        "score": score,
        "detail": f"L2 层 block×{block_n} + major×{major_n}",
    }


def score_rag_evidence(rag_result: dict = None) -> dict:
    """RAG 证据分：是否有不利条款匹配（structured_matches 命中规则）"""
    if not rag_result:
        return {"score": 70, "detail": "RAG 未执行（默认中位）"}
    matches = rag_result.get("structured_matches", [])
    # 命中准入规则（如 DTI 超标）扣分
    adverse = [m for m in matches if m.get("matched") and m.get("severity") in ("reject", "block")]
    penalty = len(adverse) * 15
    score = max(0, 90 - penalty)
    return {
        "score": score,
        "detail": f"匹配 {len(matches)} 条规则，不利 {len(adverse)} 条",
    }


def score_kg_risk(kg_alerts: list = None) -> dict:
    """知识图谱风险分：KG 告警数转化"""
    if not kg_alerts:
        return {"score": 100, "detail": "无 KG 告警"}
    high_n = sum(1 for a in kg_alerts if a.get("level") == "high")
    penalty = len(kg_alerts) * 20 + high_n * 10
    score = max(0, 100 - penalty)
    return {
        "score": score,
        "detail": f"KG 告警 {len(kg_alerts)} 条（high {high_n}）",
    }


def score_llm_semantic(inference_result: dict = None, layered_result: dict = None) -> dict:
    """LLM 语义风险分：推理风险等级 + 风险点数
    注：LLM 风险等级部分基于规则问题推断，需结合规则引擎结果做去重和降级
    - 规则引擎无 block 时，LLM high 降级为 medium（避免循环放大）
    - 过滤掉与规则引擎发现重复的风险点（避免双重扣分）
    """
    if not inference_result:
        return {"score": 70, "detail": "LLM 推理未执行"}
    risk_level = inference_result.get("risk_level", "unknown")
    risk_points = inference_result.get("risk_points", [])

    # 规则引擎无 block/major 时，LLM 的 high 降级为 medium（LLM 过度保守矫正）
    sev = (layered_result or {}).get("severity_summary", {})
    has_hard_issue = sev.get("block", 0) > 0 or sev.get("major", 0) > 0
    effective_level = risk_level
    if not has_hard_issue and risk_level == "high":
        effective_level = "medium"

    # 过滤重复风险点：LLM risk_points 中与规则引擎发现类型重复的不计
    rule_categories = set()
    for f in (layered_result or {}).get("findings", []):
        cat = f.get("category", "")
        if "欺诈" in cat or "fraud" in cat.lower():
            rule_categories.add("suspected_fraud")
        if "矛盾" in cat or "不一致" in cat:
            rule_categories.add("material_contradiction")
    filtered_rps = [rp for rp in risk_points if rp.get("type") not in rule_categories]
    rp_n = len(filtered_rps)

    level_score = {"none": 100, "low": 90, "medium": 70, "high": 45, "unknown": 70}.get(effective_level, 70)
    # 风险点数扣分（每条 -3，最多扣 25）
    penalty = min(rp_n * 3, 25)
    score = max(0, level_score - penalty)
    detail_parts = [f"风险等级 {risk_level}"]
    if effective_level != risk_level:
        detail_parts.append(f"→降级 {effective_level}（规则无 block/major）")
    detail_parts.append(f"+ {rp_n} 个有效风险点")
    if len(risk_points) != rp_n:
        detail_parts.append(f"（过滤 {len(risk_points) - rp_n} 条重复）")
    return {
        "score": score,
        "detail": " ".join(detail_parts),
    }


def score_completeness(layered_result: dict, strategy: dict = None) -> dict:
    """材料完整性分：按缺失比例扣分（缺1/8 和缺5/8 严重程度不同）"""
    l1_findings = [f for f in layered_result.get("findings", []) if f["layer"] == "L1"]
    block_n = sum(1 for f in l1_findings if f["severity"] == "block")
    major_n = sum(1 for f in l1_findings if f["severity"] == "major")

    # 按必填材料总数计算缺失比例，比例扣分更合理
    required_total = 0
    if strategy:
        required_total = sum(1 for d in strategy.get("required_docs", []) if d.get("required", True))
    missing_n = block_n + major_n
    if required_total > 0 and missing_n > 0:
        # 缺失比例 0~1，扣分 0~80（缺一半扣40，全缺扣80）
        ratio = min(missing_n / required_total, 1.0)
        score = max(0, round(100 - ratio * 80))
        detail = f"缺失 {missing_n}/{required_total} 项必填（比例 {ratio*100:.0f}%）"
    else:
        penalty = block_n * 30 + major_n * 10
        score = max(0, 100 - penalty)
        detail = f"L1 层 block×{block_n} + major×{major_n}"
    return {
        "score": score,
        "detail": detail,
    }


# ============================================================
# 综合评分
# ============================================================
def calculate_risk_score(
    layered_result: dict,
    extracted_docs: list,
    strategy: dict,
    rag_result: dict = None,
    kg_alerts: list = None,
    inference_result: dict = None,
) -> dict:
    """
    多源融合计算综合风险分数

    :return: {
        "total_score": 0-100,         # 综合分（100=最安全）
        "risk_level": "low"|"medium"|"high",
        "action": "auto_pass"|"manual_review"|"investigate"|"reject",
        "action_label": str,
        "dimension_scores": {dim: {score, detail}},
        "rule_block_override": bool,  # 规则一票否决
        "explanation": str,           # 分流理由
    }
    """
    weights = strategy.get("score_weights", {
        "completeness": 0.20, "field_logic": 0.15, "business_rule": 0.25,
        "fraud": 0.20, "ocr_confidence": 0.10, "rag_evidence": 0.05, "llm_semantic": 0.05,
    })

    # 计算各维度分数
    dim_completeness = score_completeness(layered_result, strategy)
    dim_cross_doc = score_cross_doc(layered_result)           # field_logic
    dim_rules = score_rules(layered_result)                   # business_rule + fraud 拆分
    dim_ocr = score_ocr_confidence(extracted_docs)
    dim_rag = score_rag_evidence(rag_result)
    dim_kg = score_kg_risk(kg_alerts)
    dim_llm = score_llm_semantic(inference_result, layered_result)

    # 业务规则分（L3）和欺诈分（L4）拆分
    findings = layered_result.get("findings", [])
    l3_findings = [f for f in findings if f["layer"] == "L3"]
    l4_findings = [f for f in findings if f["layer"] == "L4"]
    l3_result = {"findings": l3_findings,
                 "severity_summary": {
                     s: sum(1 for f in l3_findings if f["severity"] == s)
                     for s in ("block", "major", "minor", "info")
                 }}
    l4_result = {"findings": l4_findings,
                 "severity_summary": {
                     s: sum(1 for f in l4_findings if f["severity"] == s)
                     for s in ("block", "major", "minor", "info")
                 }}
    dim_business = score_rules(l3_result)
    dim_fraud = score_rules(l4_result)
    # KG 风险并入欺诈维度
    dim_fraud_merged = {
        "score": round((dim_fraud["score"] * 0.6 + dim_kg["score"] * 0.4)),
        "detail": dim_fraud["detail"] + " | " + dim_kg["detail"],
    }

    dimension_scores = {
        "completeness": dim_completeness,
        "field_logic": dim_cross_doc,
        "business_rule": dim_business,
        "fraud": dim_fraud_merged,
        "ocr_confidence": dim_ocr,
        "rag_evidence": dim_rag,
        "llm_semantic": dim_llm,
    }

    # 加权融合
    dim_to_weight = {
        "completeness": "completeness",
        "field_logic": "field_logic",
        "business_rule": "business_rule",
        "fraud": "fraud",
        "ocr_confidence": "ocr_confidence",
        "rag_evidence": "rag_evidence",
        "llm_semantic": "llm_semantic",
    }
    total = 0.0
    weight_sum = 0.0
    for dim, wkey in dim_to_weight.items():
        w = weights.get(wkey, 0)
        total += dimension_scores[dim]["score"] * w
        weight_sum += w
    total_score = round(total / weight_sum) if weight_sum > 0 else 50

    # ---- 三档分流 ----
    # rule_block_override：只对 L2/L3/L4 的 block 一票否决
    # L1 材料缺失允许人工补交，不一票否决（已降级为 major，此处显式排除 L1）
    all_findings = layered_result.get("findings", [])
    hard_block_findings = [
        f for f in all_findings
        if f.get("severity") == "block" and f.get("layer") in ("L2", "L3", "L4")
    ]
    rule_block_override = len(hard_block_findings) > 0

    # major 级规则发现（L2/L3/L4）：多条 major 或单条 major + 中等分数 → 强制人工复核
    # 避免 L3-02 流水月数不足 等重大风险被高分掩盖导致 auto_pass
    hard_major_findings = [
        f for f in all_findings
        if f.get("severity") == "major" and f.get("layer") in ("L2", "L3", "L4")
    ]
    hard_major_count = len(hard_major_findings)

    # LLM 语义风险点升级：当规则引擎无 findings 但 LLM 发现多个严重风险点时
    # 避免"规则盲区+LLM高分"导致的误放行（如经营贷无 income_certificate 时 L3-06 不触发）
    llm_risk_points = inference_result.get("risk_points", []) if inference_result else []
    # 注意：LLM 输出的 risk_points 使用 "level" 字段（high/medium/low），而非 "severity"
    # 见 step7_inference_risk.py prompt 定义
    llm_high_risks = [r for r in llm_risk_points if r.get("level") == "high"]
    llm_medium_risks = [r for r in llm_risk_points if r.get("level") == "medium"]
    # 含 income_mismatch 或 suspected_fraud 的 high/medium 风险点应升级
    _escalation_types = {"income_mismatch", "suspected_fraud", "material_contradiction"}
    llm_escalation_risks = [
        r for r in llm_high_risks + llm_medium_risks
        if r.get("type") in _escalation_types
    ]

    if rule_block_override:
        # 欺诈/造假类 block → 直接高风险重点调查
        risk_level = "high"
        action = "investigate"
        action_label = "重点调查"
        block_ids = [f.get("rule_id", "") for f in hard_block_findings]
        explanation = f"触发 {len(hard_block_findings)} 条一票否决规则（{','.join(block_ids)}），强制进入重点调查"
    elif hard_major_count >= 2 or (hard_major_count >= 1 and total_score < 90):
        # 多条 major 或单条 major 但分数不够高（<90）→ 强制人工复核
        risk_level = "medium"
        action = "manual_review"
        action_label = "人工复核"
        major_ids = [f.get("rule_id", "") for f in hard_major_findings]
        explanation = (f"触发 {hard_major_count} 条重大风险规则（{','.join(major_ids)}），"
                       f"综合分 {total_score}，需人工复核")
    elif len(llm_escalation_risks) >= 2 or (len(llm_high_risks) >= 1 and len(llm_escalation_risks) >= 1):
        # LLM 发现多个升级类风险点（收入不符/欺诈嫌疑/材料矛盾）→ 强制人工复核
        risk_level = "medium"
        action = "manual_review"
        action_label = "人工复核"
        risk_types = [r.get("type", "") for r in llm_escalation_risks]
        explanation = (f"LLM 语义层发现 {len(llm_escalation_risks)} 个升级类风险点"
                       f"（{','.join(risk_types)}），综合分 {total_score}，需人工复核")
    elif total_score >= 80:
        risk_level = "low"
        action = "auto_pass"
        action_label = "自动通过"
        explanation = f"综合风险分 {total_score}（≥80），各维度均表现良好，建议自动通过"
    elif total_score >= 50:
        risk_level = "medium"
        action = "manual_review"
        action_label = "人工复核"
        explanation = f"综合风险分 {total_score}（50-79），存在中等风险点，需人工复核"
    else:
        risk_level = "high"
        action = "investigate"
        action_label = "重点调查"
        explanation = f"综合风险分 {total_score}（<50），多维度风险较高，需重点调查"

    # 触发 block 一票否决时，强制将分数压到高风险区间，使分数与 action 一致
    # 避免 block 案例因其他维度满分而总分仍在 87-90 的"中高"区间
    # 按 block 数量分级，区分"单block重点调查"与"多block严重欺诈"：
    #   1 block → ≤49（重点调查）
    #   2 block → ≤39（严重欺诈嫌疑）
    #   3+ block → ≤29（极高风险）
    if rule_block_override:
        block_count = len(hard_block_findings)
        if block_count >= 3:
            display_score = min(total_score, 29)
        elif block_count == 2:
            display_score = min(total_score, 39)
        else:
            display_score = min(total_score, 49)
    else:
        display_score = total_score

    return {
        "total_score": display_score,
        "risk_level": risk_level,
        "action": action,
        "action_label": action_label,
        "dimension_scores": dimension_scores,
        "rule_block_override": rule_block_override,
        "explanation": explanation,
        # 兼容旧接口
        "total_confidence": round(total_score / 100, 3),
        "decision": "reject" if action == "investigate" and rule_block_override else (
            "auto_pass" if action == "auto_pass" else "manual_review"
        ),
    }


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    from business_entry import detect_product, load_strategy
    from layered_rules import run_all_layers

    # 模拟 12-谢衍 的材料（多身份证号 + USDT + 逾期）
    docs = [
        {"doc_type": "business_license", "fields": {"expiry_date": "2009-12-31"}, "file": "license.jpg"},
        {"doc_type": "id_card_front", "fields": {"id_number": "79175219700809243X", "name": "谢衍"},
         "confidence": 0.98, "file": "id1.jpg"},
        {"doc_type": "loan_application", "fields": {"id_number": "791752197108091901", "name": "谢衍",
                                                      "spouse_id_number": "861453198405102155"},
         "confidence": 0.99, "file": "app.jpg"},
        {"doc_type": "marriage_cert", "fields": {"spouse_id_number": "861453198805101557"},
         "confidence": 0.98, "file": "marry.jpg"},
        {"doc_type": "bank_statement", "fields": {}, "ocr_text": "USDT 充值 5000",
         "transactions": [{"date": "2025-01-01"}], "confidence": 0.95},
        {"doc_type": "income_certificate", "fields": {"monthly_income": "8000"}, "confidence": 0.98},
        {"doc_type": "credit_report", "fields": {"overdue_total": 8, "overdue_consecutive": 3}, "confidence": 0.98},
    ]
    strat = detect_product(docs)["strategy"]
    layered = run_all_layers(docs, strat, rule_result={"avg_monthly_income": 5000}, kg_alerts=[])

    score = calculate_risk_score(
        layered, docs, strat,
        rag_result={"structured_matches": [{"matched": True, "severity": "reject"}]},
        kg_alerts=[{"level": "high", "evidence": "test"}],
        inference_result={"risk_level": "high", "risk_points": [{"type": "test"}, {"type": "test2"}]},
    )

    print(f"综合风险分: {score['total_score']}  风险等级: {score['risk_level']}")
    print(f"动作: {score['action']} ({score['action_label']})")
    print(f"规则一票否决: {score['rule_block_override']}")
    print(f"理由: {score['explanation']}")
    print(f"\n各维度评分:")
    for dim, d in score["dimension_scores"].items():
        print(f"  {dim:16s}: {d['score']:3d}  ({d['detail']})")

    print("\n" + "=" * 50)
    # 测试低风险场景（材料齐全、无问题）
    clean_docs = [
        {"doc_type": "id_card_front", "fields": {"id_number": "110101199001011234", "name": "张三"}, "confidence": 0.98},
        {"doc_type": "id_card_back", "fields": {}, "confidence": 0.95},
        {"doc_type": "bank_statement", "fields": {}, "transactions": [{"date": f"2025-0{i}-15"} for i in range(1, 7)], "confidence": 0.97},
        {"doc_type": "income_certificate", "fields": {"monthly_income": "15000"}, "confidence": 0.98},
        {"doc_type": "credit_report", "fields": {"overdue_total": 0, "overdue_consecutive": 0}, "confidence": 0.98},
    ]
    strat2 = load_strategy("personal_consumer")
    layered2 = run_all_layers(clean_docs, strat2)
    score2 = calculate_risk_score(layered2, clean_docs, strat2)
    print(f"低风险场景: 综合分 {score2['total_score']}  动作: {score2['action_label']}  理由: {score2['explanation']}")
