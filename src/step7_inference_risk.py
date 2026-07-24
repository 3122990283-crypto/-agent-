"""
⑦ 推理模型层（语义风险）
只做概率性、语义类判断，不做数学计算
- 流水异常模式识别（Isolation Forest + LLM 语义判断）
- 经营痕迹识别（企业贷）
- 贷款用途合理性
"""
import json
import numpy as np
from sklearn.ensemble import IsolationForest

from api_clients import get_qwen


# ============================================================
# 流水时序特征工程
# ============================================================
def extract_flow_features(transactions: list) -> dict:
    """提取流水时序特征"""
    if not transactions:
        return {}

    incomes = [float(t.get("income") or 0) for t in transactions]
    expenses = [float(t.get("expense") or 0) for t in transactions]
    balances = [float(t.get("balance") or 0) for t in transactions]
    dates = [t.get("date", "") for t in transactions]

    # 月均
    avg_income = np.mean(incomes) if incomes else 0
    avg_expense = np.mean(expenses) if expenses else 0

    # 变异系数 CV（规律性）
    income_cv = np.std(incomes) / np.mean(incomes) if np.mean(incomes) > 0 else 0

    # 大额整数交易占比
    large_int_count = sum(1 for x in incomes if x > 10000 and x == int(x))
    large_int_ratio = large_int_count / max(len(incomes), 1)

    # 快进快出（入账后余额很快又转出，简化判断：同日大额收支）
    quick_in_out = 0
    for i in range(len(transactions) - 1):
        if incomes[i] > 5000 and abs(incomes[i] - expenses[i + 1]) / max(incomes[i], 1) < 0.1:
            quick_in_out += 1
    quick_in_out_ratio = quick_in_out / max(len(incomes), 1)

    # 夜间交易占比（22:00-06:00，如果有时间字段）
    night_count = sum(1 for d in dates if " " in d and d.split(" ")[1][:2] in ["22", "23", "00", "01", "02", "03", "04", "05"])
    night_ratio = night_count / max(len(dates), 1)

    # 过渡性资金（进账即全额转出）
    transit_count = 0
    for i in range(len(transactions) - 1):
        if incomes[i] > 0 and abs(incomes[i] - expenses[i + 1]) / max(incomes[i], 1) < 0.05:
            transit_count += 1
    transit_ratio = transit_count / max(len(incomes), 1)

    return {
        "total_transactions": len(transactions),
        "avg_income": round(float(avg_income), 2),
        "avg_expense": round(float(avg_expense), 2),
        "income_cv": round(float(income_cv), 4),
        "large_int_ratio": round(float(large_int_ratio), 4),
        "quick_in_out_ratio": round(float(quick_in_out_ratio), 4),
        "night_ratio": round(float(night_ratio), 4),
        "transit_ratio": round(float(transit_ratio), 4),
    }


def detect_flow_anomaly(features: dict) -> dict:
    """
    流水异常检测（规则 + Isolation Forest 占位）
    由于单条样本无法用 Isolation Forest，这里用规则判断
    """
    warnings = []

    if features.get("large_int_ratio", 0) > 0.3:
        warnings.append(f"大额整数交易占比 {features['large_int_ratio']:.0%}，疑似整存整取")

    if features.get("quick_in_out_ratio", 0) > 0.2:
        warnings.append(f"快进快出占比 {features['quick_in_out_ratio']:.0%}，疑似过渡性资金")

    if features.get("transit_ratio", 0) > 0.2:
        warnings.append(f"过渡性资金占比 {features['transit_ratio']:.0%}，进账即转出")

    if features.get("night_ratio", 0) > 0.1:
        warnings.append(f"夜间交易占比 {features['night_ratio']:.0%}，交易时间异常")

    if features.get("income_cv", 0) > 1.5:
        warnings.append(f"入账变异系数 {features['income_cv']:.2f}，入账极不规律")

    return {
        "anomaly": len(warnings) > 0,
        "warnings": warnings,
        "features": features,
    }


# ============================================================
# LLM 语义风险判断
# ============================================================
RISK_SYSTEM_PROMPT = """你是信贷风控专家，擅长识别贷款材料中的语义风险。
你的任务是：根据给定的材料摘要和规则引擎结果，判断是否存在风险。

【重要原则】
1. 只识别规则引擎未覆盖的语义风险，不要重复规则引擎已发现的问题
2. 规则引擎结果为"无"时，默认风险等级为 none/low，除非发现明确语义风险
3. 数值差异在 5% 以内视为 OCR 误差，不视为风险
4. 银行流水材料有 OCR 文本但无结构化交易明细时，不算"流水缺失"（OCR 文本已含信息）
5. 无明确证据不要臆测风险，宁可漏报也不要误报

风险类型（仅当有明确证据时才标注）：
1. income_mismatch: 收入与流水/税务明显不一致（差异>20%）
2. cash_flow_anomaly: 流水异常模式（整存整取、快进快出、过渡性资金等）
3. business_trace_missing: 企业流水缺少经营痕迹（无工资、税金、上下游往来）
4. purpose_unreasonable: 贷款用途不合理
5. material_contradiction: 多份材料相互明显矛盾
6. suspected_fraud: 疑似造假（需有具体证据）

风险等级判断标准：
- none: 无任何风险点
- low: 仅有 1-2 个低风险点，不影响审批
- medium: 存在中等风险点，需人工核实
- high: 存在明确造假或重大矛盾证据

输出格式（JSON）：
{
  "risk_level": "high/medium/low/none",
  "risk_points": [
    {"type": "风险类型", "level": "high/medium/low", "reason": "具体原因"}
  ],
  "suggestion": "处理建议"
}
"""


def assess_semantic_risk(extracted_docs: list, rule_issues: list, cross_doc_issues: list) -> dict:
    """
    推理模型语义风险评估
    :param extracted_docs: step2 输出
    :param rule_issues: step6 规则引擎问题
    :param cross_doc_issues: step4 一致性问题
    :return: LLM 风险评估结果
    """
    # ---- 准备材料摘要 ----
    summaries = []
    flow_features = None
    flow_anomaly = None

    for doc in extracted_docs:
        doc_type = doc.get("doc_type")
        fields = doc.get("fields", {})
        transactions = doc.get("transactions", [])

        summary = f"[{doc_type}]"
        for k, v in fields.items():
            if v is not None:
                summary += f" {k}={v}"

        if doc_type == "bank_statement" and transactions:
            flow_features = extract_flow_features(transactions)
            flow_anomaly = detect_flow_anomaly(flow_features)
            summary += f"\n  流水特征: {json.dumps(flow_features, ensure_ascii=False)}"
            if flow_anomaly["anomaly"]:
                summary += f"\n  异常告警: {'; '.join(flow_anomaly['warnings'])}"
            # 交易明细前 20 条
            tx_str = "\n".join(
                f"    {t.get('date','')} {t.get('desc','')} 收{t.get('income',0)} 支{t.get('expense',0)} 余{t.get('balance',0)}"
                for t in transactions[:20]
            )
            summary += f"\n  交易明细(前20条):\n{tx_str}"

        summaries.append(summary)

    # ---- 准备规则引擎 + 一致性问题 ----
    rule_str = "\n".join(f"- [{i['level']}] {i.get('msg','')}" for i in rule_issues) or "无"
    cross_str = "\n".join(f"- [{i['level']}] {i.get('msg','')}" for i in cross_doc_issues) or "无"

    # ---- 调用 LLM ----
    prompt = f"""请评估以下贷款申请的风险：

【材料摘要】
{chr(10).join(summaries)}

【规则引擎结果】
{rule_str}

【跨文档一致性结果】
{cross_str}

【流水异常检测结果】
{json.dumps(flow_anomaly, ensure_ascii=False) if flow_anomaly else '无流水数据'}

请综合判断风险等级，并指出具体风险点。严格按 JSON 格式输出。"""

    print(f"\n[Step7] 调用千问 LLM 评估语义风险")
    try:
        result = get_qwen().chat_json(prompt, system=RISK_SYSTEM_PROMPT, temperature=0.2)
    except json.JSONDecodeError as e:
        print(f"[Step7] LLM 输出解析失败: {e}")
        result = {
            "risk_level": "unknown",
            "risk_points": [],
            "suggestion": f"LLM 输出解析失败: {e}",
        }

    # 补充流水异常检测结果
    if flow_anomaly and flow_anomaly["anomaly"]:
        result.setdefault("risk_points", []).append({
            "type": "cash_flow_anomaly",
            "level": "medium",
            "reason": "; ".join(flow_anomaly["warnings"]),
        })

    return result
