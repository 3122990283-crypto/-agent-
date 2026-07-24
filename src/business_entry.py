"""
业务入口模块（business_entry）

职责：
1. 贷款产品识别 — 根据申请材料 + 申请表 LLM 识别产品类型
2. 策略加载 — 按产品加载审核策略（必备材料/规则阈值/评分权重/风险驱动检索配置）

这是整个 Agent 的第一步：先识别"审什么"，再决定"怎么审"。
替代原来 demo_batch.py 硬编码 loan_product 参数的做法。
"""
import json
from typing import Optional

import config
from api_clients import get_qwen


# ============================================================
# 产品审核策略库
# 每个产品定义：必备材料 / 选填材料 / 业务规则阈值 / 评分权重 / 风险检索配置
# ============================================================
PRODUCT_STRATEGIES = {
    # ---------- 个人消费贷 ----------
    "personal_consumer": {
        "name": "个人消费贷",
        "category": "retail",
        "description": "个人用于消费用途的无抵押贷款",
        "required_docs": [
            {"doc_type": "id_card_front", "required": True, "label": "身份证正面"},
            {"doc_type": "id_card_back", "required": True, "label": "身份证反面"},
            {"doc_type": "bank_statement", "required": True, "label": "银行流水（近6个月）"},
            {"doc_type": "income_certificate", "required": True, "label": "收入证明"},
            {"doc_type": "loan_investigation", "required": True, "label": "个人贷款调查报告"},
        ],
        "optional_docs": [
            {"doc_type": "marriage_cert", "label": "婚姻证明"},
            {"doc_type": "divorce_cert", "label": "离婚证"},
            {"doc_type": "household_register_front", "label": "户口本（户主页）"},
            {"doc_type": "household_register_self", "label": "户口本（本人页）"},
            {"doc_type": "bank_card", "label": "银行卡"},
            {"doc_type": "social_security", "label": "社会保险参保证明"},
            {"doc_type": "tax_certificate", "label": "个人所得税完税证明"},
            {"doc_type": "loan_application", "label": "贷款申请表"},
            {"doc_type": "loan_settlement", "label": "贷款结清证明"},
        ],
        # 业务规则阈值（覆盖全局默认）
        "thresholds": {
            "dti_warning": 0.5,                 # 个人 DTI > 50% 预警
            "dti_reject": 0.7,                 # 个人 DTI > 70% 拒贷
            "income_mismatch": 0.2,            # 收入差异 > 20% 中风险
            "credit_report_fresh_days": 15,
            "overdue_total_threshold": 6,
            "overdue_consecutive_threshold": 3,
            "flow_month_min": 6,               # 流水至少 6 个月
            "min_loan_amount": 10000,
            "max_loan_amount": 500000,
        },
        # 评分权重（该产品的风险评分侧重）
        "score_weights": {
            "completeness": 0.20,   # 材料完整性权重高（消费贷看材料齐全）
            "field_logic": 0.15,
            "business_rule": 0.25,
            "fraud": 0.20,
            "ocr_confidence": 0.10,
            "rag_evidence": 0.05,
            "llm_semantic": 0.05,
        },
        # 风险驱动检索配置：发现某类异常时主动检索的规则关键词
        "risk_search_triggers": {
            "income_mismatch": ["收入认定标准", "流水月均入账核定", "收入证明真实性核查"],
            "overdue_excess": ["征信逾期容忍度", "个人贷款逾期次数准入"],
            "id_mismatch": ["身份核实", "客户身份识别", "实名认证"],
            "high_dti": ["偿债能力评估", "债务收入比", "还款能力"],
        },
    },

    # ---------- 企业经营贷 ----------
    "business_loan": {
        "name": "企业经营贷",
        "category": "corporate",
        "description": "面向中小企业的经营性贷款",
        "required_docs": [
            {"doc_type": "business_license", "required": True, "label": "营业执照"},
            {"doc_type": "id_card_front", "required": True, "label": "法人身份证正面"},
            {"doc_type": "id_card_back", "required": True, "label": "法人身份证反面"},
            {"doc_type": "bank_statement", "required": True, "label": "企业银行流水（近12个月）"},
            {"doc_type": "fund_flow_analysis", "required": True, "label": "资金流水分析表"},
            {"doc_type": "loan_investigation", "required": True, "label": "个人贷款调查报告"},
        ],
        "optional_docs": [
            {"doc_type": "tax_certificate", "label": "个人所得税完税证明"},
            {"doc_type": "food_license", "label": "食品经营许可证"},
            {"doc_type": "tobacco_license", "label": "烟草专卖零售许可证"},
            {"doc_type": "transport_license", "label": "道路运输许可证"},
            {"doc_type": "real_estate_cert", "label": "不动产权证（抵押物）"},
            {"doc_type": "real_estate_query", "label": "不动产信息查询结果"},
            {"doc_type": "social_security", "label": "社会保险参保证明"},
            {"doc_type": "household_register_front", "label": "户口本（户主页）"},
            {"doc_type": "household_register_self", "label": "户口本（本人页）"},
            {"doc_type": "bank_card", "label": "银行卡"},
            {"doc_type": "loan_application", "label": "贷款申请表"},
            {"doc_type": "loan_settlement", "label": "贷款结清证明"},
        ],
        "thresholds": {
            "dti_warning": 0.6,
            "dti_reject": 0.8,
            "income_mismatch": 0.3,            # 企业收入波动大，阈值放宽
            "credit_report_fresh_days": 30,   # 企业征信 30 天
            "overdue_total_threshold": 8,
            "overdue_consecutive_threshold": 3,
            "flow_month_min": 12,              # 企业流水至少 12 个月
            "business_year_min": 2,           # 经营满 2 年
            "min_loan_amount": 100000,
            "max_loan_amount": 5000000,
        },
        "score_weights": {
            "completeness": 0.15,
            "field_logic": 0.15,
            "business_rule": 0.25,            # 企业贷看经营指标
            "fraud": 0.20,
            "ocr_confidence": 0.05,
            "rag_evidence": 0.10,             # 企业贷政策依据更重要
            "llm_semantic": 0.10,
        },
        "risk_search_triggers": {
            "license_expired": ["经营资质有效性", "营业执照年检", "行政许可延续"],
            "business_age_short": ["企业经营年限准入", "新设企业贷款风险"],
            "income_mismatch": ["企业经营收入核定", "流水与报表一致性"],
            "overdue_excess": ["企业征信逾期", "对公贷款不良记录"],
        },
    },

    # ---------- 个人住房贷款 ----------
    "mortgage": {
        "name": "个人住房贷款",
        "category": "retail",
        "description": "个人购买住房的抵押贷款",
        "required_docs": [
            {"doc_type": "id_card_front", "required": True, "label": "身份证正面"},
            {"doc_type": "id_card_back", "required": True, "label": "身份证反面"},
            {"doc_type": "bank_statement", "required": True, "label": "银行流水（近6个月）"},
            {"doc_type": "income_certificate", "required": True, "label": "收入证明"},
            {"doc_type": "loan_investigation", "required": True, "label": "个人贷款调查报告"},
            {"doc_type": "real_estate_query", "required": True, "label": "不动产信息查询结果"},
            {"doc_type": "real_estate_cert", "required": True, "label": "不动产权证/购房合同"},
            {"doc_type": "marriage_cert", "required": True, "label": "婚姻证明（结婚证或离婚证）"},
        ],
        "optional_docs": [
            {"doc_type": "divorce_cert", "label": "离婚证"},
            {"doc_type": "household_register_front", "label": "户口本（户主页）"},
            {"doc_type": "household_register_self", "label": "户口本（本人页）"},
            {"doc_type": "bank_card", "label": "银行卡"},
            {"doc_type": "social_security", "label": "社会保险参保证明"},
            {"doc_type": "tax_certificate", "label": "个人所得税完税证明"},
            {"doc_type": "loan_application", "label": "贷款申请表"},
            {"doc_type": "loan_settlement", "label": "贷款结清证明"},
            {"doc_type": "fund_flow_analysis", "label": "资金流水分析表"},
        ],
        "thresholds": {
            "dti_warning": 0.5,
            "dti_reject": 0.6,                # 房贷 DTI 管控严
            "income_mismatch": 0.2,
            "credit_report_fresh_days": 15,
            "overdue_total_threshold": 6,
            "overdue_consecutive_threshold": 2,
            "flow_month_min": 6,
            "min_loan_amount": 100000,
            "max_loan_amount": 10000000,
        },
        "score_weights": {
            "completeness": 0.20,
            "field_logic": 0.20,              # 房贷看产权一致性
            "business_rule": 0.20,
            "fraud": 0.15,
            "ocr_confidence": 0.05,
            "rag_evidence": 0.10,
            "llm_semantic": 0.10,
        },
        "risk_search_triggers": {
            "property_dispute": ["不动产抵押", "产权核查", "押品评估"],
            "income_mismatch": ["还款能力", "收入负债比", "房贷偿债能力"],
            "marriage_status": ["共同还款人", "婚姻状况核实", "夫妻共同债务"],
        },
    },

    # ---------- 个人汽车贷款 ----------
    # 已移除：数据集无驾驶证/购车合同材料，无法支持
}


# 兼容：保留 auto_loan key 但指向 personal_consumer（防止旧代码引用报错）
PRODUCT_STRATEGIES["auto_loan"] = PRODUCT_STRATEGIES["personal_consumer"]


# ============================================================
# 产品识别
# ============================================================
def detect_product(extracted_docs: list, loan_application_text: str = "") -> dict:
    """
    识别贷款产品类型。
    优先级：
    1. 申请表 OCR 文本中显式标注的产品类型
    2. 材料组合特征推断（有营业执照→经营贷，有不动产→房贷）
    3. LLM 兜底识别

    :param extracted_docs: step2 抽取后的文档列表
    :param loan_application_text: 申请表 OCR 文本（可选，最准）
    :return: {
        "product_key": str,
        "product_name": str,
        "confidence": float,
        "evidence": str,        # 识别依据
        "strategy": dict,       # 加载的策略
    }
    """
    doc_types = [d.get("doc_type", "") for d in extracted_docs if d.get("doc_type")]

    # ---- 1. 申请表显式标注 ----
    if loan_application_text:
        text = loan_application_text
        if any(k in text for k in ["经营贷", "企业经营", "流动资金"]):
            return _build_result("business_loan", 0.95, "申请表标注: 经营贷")
        if any(k in text for k in ["住房贷款", "房贷", "按揭", "购房"]):
            return _build_result("mortgage", 0.95, "申请表标注: 房贷")
        if any(k in text for k in ["汽车贷款", "车贷", "购车"]):
            return _build_result("auto_loan", 0.95, "申请表标注: 车贷")
        if any(k in text for k in ["消费贷", "个人消费", "信用贷"]):
            return _build_result("personal_consumer", 0.95, "申请表标注: 消费贷")

    # ---- 2. 材料组合特征推断（高置信度，无需 LLM）----
    has_business = "business_license" in doc_types or "food_license" in doc_types
    has_real_estate = "real_estate_query" in doc_types or "real_estate_cert" in doc_types
    has_financial_stmt = "financial_statement" in doc_types
    has_fund_flow = "fund_flow_analysis" in doc_types

    if has_business and (has_financial_stmt or has_fund_flow):
        return _build_result("business_loan", 0.90,
                             "材料组合含营业执照+财务报表/资金流水分析 → 经营贷")

    if has_real_estate and "marriage_cert" in doc_types:
        return _build_result("mortgage", 0.85,
                             "材料组合含不动产+婚姻证明 → 房贷")

    # ---- 3. 申请表 LLM 识别（兜底）----
    if loan_application_text and len(loan_application_text) > 50:
        try:
            result = _llm_detect_product(loan_application_text)
            if result and result["product_key"] in PRODUCT_STRATEGIES:
                return _build_result(result["product_key"], result["confidence"],
                                     f"LLM 识别: {result['evidence']}")
        except Exception as e:
            print(f"[业务入口] LLM 产品识别失败: {e}")

    # ---- 4. 默认消费贷（材料最少能匹配）----
    return _build_result("personal_consumer", 0.50,
                         "未明确标注，默认按个人消费贷处理（需人工确认）")


def _build_result(product_key: str, confidence: float, evidence: str) -> dict:
    strategy = PRODUCT_STRATEGIES.get(product_key, PRODUCT_STRATEGIES["personal_consumer"])
    return {
        "product_key": product_key,
        "product_name": strategy["name"],
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "strategy": strategy,
    }


def _llm_detect_product(application_text: str) -> Optional[dict]:
    """LLM 兜底识别产品类型"""
    prompt = f"""请根据贷款申请表内容，识别贷款产品类型。

申请表内容：
{application_text[:2000]}

可选产品类型：
- personal_consumer: 个人消费贷
- business_loan: 企业经营贷
- mortgage: 个人住房贷款
- auto_loan: 个人汽车贷款

请返回 JSON：
{{"product_key": "...", "confidence": 0.0-1.0, "evidence": "识别依据（一句话）"}}"""

    resp = get_qwen().chat_json(prompt, temperature=0.1)
    if resp and "product_key" in resp:
        return resp
    return None


# ============================================================
# 策略加载
# ============================================================
def load_strategy(product_key: str) -> dict:
    """加载产品审核策略"""
    return PRODUCT_STRATEGIES.get(product_key, PRODUCT_STRATEGIES["personal_consumer"])


def list_products() -> list:
    """列出所有产品（供前端/上传门户使用）"""
    _SKIP = {"auto_loan"}  # 兼容键，不展示
    return [
        {
            "key": k,
            "name": v["name"],
            "category": v["category"],
            "description": v["description"],
            "required_docs": [d["label"] for d in v["required_docs"] if d.get("required", True)],
            "optional_docs": [d["label"] for d in v["optional_docs"]],
        }
        for k, v in PRODUCT_STRATEGIES.items()
        if k not in _SKIP
    ]


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    # 模拟 12-谢衍 的材料组合（有营业执照+财务相关）
    docs = [
        {"doc_type": "business_license"},
        {"doc_type": "id_card_front"},
        {"doc_type": "bank_statement"},
        {"doc_type": "fund_flow_analysis"},
        {"doc_type": "loan_application"},
    ]
    r = detect_product(docs)
    print(f"产品识别: {r['product_name']} (conf={r['confidence']})")
    print(f"依据: {r['evidence']}")
    print(f"\n策略 - 必填材料:")
    for d in r["strategy"]["required_docs"]:
        print(f"  {'✓' if d['required'] else '○'} {d['label']}")
    print(f"\n策略 - 评分权重:")
    for k, v in r["strategy"]["score_weights"].items():
        print(f"  {k}: {v}")
    print(f"\n策略 - 风险检索触发:")
    for trigger, kws in r["strategy"]["risk_search_triggers"].items():
        print(f"  {trigger} → {kws}")

    print("\n" + "=" * 40)
    print("全部产品:")
    for p in list_products():
        print(f"  {p['key']}: {p['name']} ({p['category']})")
