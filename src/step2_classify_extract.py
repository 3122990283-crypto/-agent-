"""
② 文档分类 + OCR + 字段抽取层（PaddleOCR-VL 1.6 API 一步到位）

流程：
1. 调 PaddleOCR-VL 1.6 API 拿到 markdown 全文
2. 用千问 LLM 做 doc_type 分类 + 字段 KV 抽取（一次调用）
3. 输出结构化字段 JSON
"""
import json
from api_clients import get_paddle_ocr, get_qwen


# ============================================================
# doc_type 枚举白名单（支持 FCMBench 21 类 + 原有类）
# ============================================================
DOC_TYPES = [
    # 身份与银行
    "id_card_front",              # 身份证（人像面/正面）
    "id_card_back",               # 身份证（国徽面/反面）
    "bank_card",                  # 银行卡
    # 收入与流水
    "bank_statement",             # 流水明细
    "income_certificate",         # 收入证明
    "fund_flow_analysis",         # 资金流水分析表
    # 企业资质
    "business_license",           # 企业法人营业执照
    "tax_certificate",            # 个人所得税完税证明
    "social_security",            # 社会保险参保证明
    # 户籍与婚姻
    "household_register_front",   # 户口本（户主页）
    "household_register_self",    # 户口本（本人页）
    "marriage_cert",              # 结婚证（电子版）
    "divorce_cert",               # 离婚证（电子版）
    # 不动产
    "real_estate_cert",           # 不动产权证书
    "real_estate_query",          # 不动产信息查询结果
    # 贷款相关
    "loan_application",           # 贷款申请表
    "loan_settlement",            # 贷款结清证明
    "loan_investigation",         # 个人贷款调查报告
    # 行业许可证
    "tobacco_license",            # 烟草专卖零售许可证
    "transport_license",          # 道路运输许可证
    "food_license",               # 食品经营许可证
    # 原有补充
    "credit_report",              # 征信报告
    "financial_statement",        # 财务报表
    "residence_proof",            # 居住证明
    "unknown",
]

# FCMBench 中文名 → doc_type 映射（用于评测脚本对齐）
FCMBENCH_TO_DOC_TYPE = {
    "身份证（人像面）": "id_card_front",
    "身份证（国徽面）": "id_card_back",
    "银行卡": "bank_card",
    "流水明细": "bank_statement",
    "收入证明": "income_certificate",
    "企业法人营业执照": "business_license",
    "户口本（户主页）": "household_register_front",
    "户口本（本人页）": "household_register_self",
    "不动产权证书": "real_estate_cert",
    "不动产信息查询结果": "real_estate_query",
    "个人所得税完税证明": "tax_certificate",
    "社会保险参保证明": "social_security",
    "结婚证（电子版）": "marriage_cert",
    "离婚证（电子版）": "divorce_cert",
    "贷款申请表": "loan_application",
    "贷款结清证明": "loan_settlement",
    "个人贷款调查报告": "loan_investigation",
    "资金流水分析表": "fund_flow_analysis",
    "烟草专卖零售许可证": "tobacco_license",
    "道路运输许可证": "transport_license",
    "食品经营许可证": "food_license",
}

# 子版式枚举（银行流水细分）
SUB_TEMPLATES = {
    "bank_statement": ["icbc", "ccb", "cmb", "boc", "abc", "spdb", "other"],
}


# ============================================================
# 各 doc_type 的字段 Schema（用于约束 LLM 输出）
# ============================================================
FIELD_SCHEMAS = {
    "id_card_front": {
        "name": "姓名",
        "id_no": "身份证号",
        "gender": "性别",
        "ethnic": "民族",
        "birth": "出生日期(YYYY-MM-DD)",
        "address": "住址",
        "valid_until": "有效期至(YYYY-MM-DD 或 长期)",
    },
    "id_card_back": {
        "issued_by": "签发机关",
        "valid_from": "有效期开始(YYYY-MM-DD)",
        "valid_until": "有效期至(YYYY-MM-DD)",
    },
    "bank_statement": {
        "account_holder": "户名",
        "account_no": "账号",
        "bank": "开户银行",
        "period": "流水期间",
    },
    "business_license": {
        "company_name": "企业名称",
        "uscc": "统一社会信用代码",
        "legal_person": "法定代表人",
        "registered_capital": "注册资本",
        "establish_date": "成立日期(YYYY-MM-DD)",
        "business_scope": "经营范围",
        "valid_until": "营业期限(YYYY-MM-DD 或 长期)",
    },
    "credit_report": {
        "report_no": "报告编号",
        "report_date": "报告时间(YYYY-MM-DD)",
        "name": "姓名",
        "id_no": "证件号码",
        "overdue_count_2y": "近2年逾期次数",
        "current_overdue": "当前逾期数",
        "five_level_classification": "五级分类",
    },
    "financial_statement": {
        "company_name": "企业名称",
        "period": "报表期间",
        "total_assets": "总资产",
        "total_liabilities": "总负债",
        "total_equity": "所有者权益",
        "net_profit": "净利润",
        "operating_income": "营业收入",
        "cash_net_increase": "现金净增加额",
        "cash_end": "期末货币资金",
        "cash_begin": "期初货币资金",
    },
    "income_certificate": {
        "name": "姓名",
        "company": "单位名称",
        "monthly_income": "月收入(元)",
        "annual_income": "年收入(元)",
        "issue_date": "开具日期(YYYY-MM-DD)",
    },
    "residence_proof": {
        "name": "姓名",
        "address": "居住地址",
        "issue_date": "开具日期(YYYY-MM-DD)",
    },
    "tax_certificate": {
        "company_name": "企业名称",
        "tax_amount": "纳税金额(元)",
        "period": "纳税期间",
    },
    "fund_flow_analysis": {
        "account_name": "账户名称",
        "account_no": "账号",
        "period": "流水期间",
        "inflow_count": "流入笔数",
        "inflow_amount": "流入金额(元)",
        "monthly_avg_inflow": "月均流入(元)",
    },
}


# ============================================================
# LLM 分类 + 字段抽取 Prompt
# ============================================================
SYSTEM_PROMPT = """你是信贷材料预审的文档解析专家。
你的任务：
1. 判断输入文本属于哪种信贷材料类型（doc_type）
2. 抽取该材料类型对应的关键字段
3. 如果是银行流水，还要还原交易明细
4. 必须严格按 JSON Schema 输出，不要输出其他内容

doc_type 枚举：
- id_card_front: 身份证（人像面/正面），含姓名/性别/民族/出生/住址/身份证号/有效期
- id_card_back: 身份证（国徽面/反面），含签发机关/有效期
- bank_card: 银行卡，含卡号/银行名/持卡人
- bank_statement: 流水明细/银行流水/交易明细，含户名/账号/开户银行/起止日期(period)/交易明细(transactions)
- income_certificate: 收入证明，含月收入/年收入/单位
- fund_flow_analysis: 资金流水分析表
- business_license: 企业法人营业执照，含统一社会信用代码/法人/注册资本
- tax_certificate: 个人所得税完税证明
- social_security: 社会保险参保证明
- household_register_front: 户口本（户主页）
- household_register_self: 户口本（本人页）
- marriage_cert: 结婚证（电子版）
- divorce_cert: 离婚证（电子版）
- real_estate_cert: 不动产权证书
- real_estate_query: 不动产信息查询结果
- loan_application: 贷款申请表
- loan_settlement: 贷款结清证明
- loan_investigation: 个人贷款调查报告
- tobacco_license: 烟草专卖零售许可证
- transport_license: 道路运输许可证
- food_license: 食品经营许可证
- credit_report: 征信报告/信用报告
- financial_statement: 财务报表（资产负债表/利润表/现金流量表）
- residence_proof: 居住证明
- unknown: 无法识别

易混类提醒：
- 身份证正面有人像和"中华人民共和国居民身份证"字样，反面是国徽
- 收入证明 vs 工作证明：收入证明含具体金额，工作证明只有职务
- 个人流水 vs 企业流水：户名是个人还是企业
- 不动产权证书 vs 不动产信息查询结果：前者是证书，后者是查询结果单
- 银行卡 vs 银行流水（关键区分）：
  * 银行卡：文本极短（<200字），只有卡号/持卡人/银行名，无日期序列、无交易记录
  * 银行流水：文本长（通常>1000字），含"交易日期/摘要/金额/余额"等多行交易记录
  * 判断规则：OCR 文本 < 300 字 且 无"余额/交易/借方/贷方"关键词 → bank_card
  * 判断规则：OCR 文本 > 500 字 或 含多行交易记录 → bank_statement
- 户口本户主页 vs 本人页：户主页有"户主"字样和户号，本人页是个人页
"""


def _hard_rule_preclassify(ocr_text: str) -> str:
    """
    硬规则预判：基于文本特征快速识别明显材料类型
    返回 doc_type 或 None（None 表示走 LLM）
    用于拦截 LLM 易错的边界场景（如银行卡 vs 流水）
    """
    text = ocr_text.strip()
    text_len = len(text)

    # 银行卡：文本极短 + 含卡号特征 + 无交易记录关键词
    has_transaction_kw = any(
        kw in text for kw in ["交易日期", "摘要", "借方", "贷方", "余额", "Transaction", "Date", "Balance"]
    )
    has_card_kw = any(kw in text for kw in ["银联", "UnionPay", "VISA", "MasterCard", "信用卡", "借记卡"])
    # 卡号特征：4 位一组，16~19 位
    import re
    card_pattern = re.findall(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", text)
    if text_len < 300 and not has_transaction_kw and (has_card_kw or len(card_pattern) > 0):
        return "bank_card"

    return None


def _build_extract_prompt(ocr_text: str) -> str:
    """构建字段抽取 prompt"""
    # 银行流水等长文档 OCR 可达 10000+ 字，截断到 6000 会导致后几个月交易丢失，
    # 引发 L3-02（月数不足）和 L3-06（收入字段缺失）连锁误判。
    # Qwen 支持 32K 上下文，15000 字（≈8000 tokens）安全且能覆盖完整流水。
    max_chars = 15000
    return f"""请解析以下 OCR 识别文本，判断材料类型并抽取字段。

OCR 文本：
---
{ocr_text[:max_chars]}
---

输出格式（严格遵守 JSON Schema）：
```json
{{
  "doc_type": "从枚举中选择",
  "sub_template": "如果是 bank_statement，填 icbc/ccb/cmb/boc/abc/spdb/other 之一；否则填 null",
  "confidence": 0.0到1.0的置信度,
  "fields": {{
    // 按 doc_type 对应的字段抽取，未识别到的字段填 null
  }},
  "transactions": [
    // 仅 bank_statement 需要：每条交易一行
    // {{"date": "YYYY-MM-DD", "desc": "摘要", "income": 0.0, "expense": 0.0, "balance": 0.0}}
    // 没有 transactions 的材料填空数组 []
  ],
  "notes": "其他需要说明的问题（如字迹模糊、关键字段缺失等），无则填空字符串"
}}
```

要求：
1. doc_type 必须从枚举选，不能自由发挥
2. 字段值尽量从原文抽取，不要编造
3. 金额必须是数字，不要带单位
4. 日期统一 YYYY-MM-DD 格式
5. 识别不出的字段填 null
6. 只输出 JSON，不要输出其他文字

银行流水 transactions 解析要点（重要）：
- 个人流水"本次金额"列：含 + 号或正数 → income（收入）；含 - 号或负数 → expense（支出）
  示例："+8000.00" → income=8000.0, expense=0.0
  示例："-1500.00" → income=0.0, expense=1500.0
- 企业流水"借方金额"列有值 → income；"贷方金额"列有值 → expense
- "工资"/"销售款"/"货款" 等收入类摘要 → income 必须有值
- "消费"/"ATM取款"/"采购款" 等支出类摘要 → expense 必须有值
- 不要把所有交易都填 income=0.0，工资/收入类必须有 income 值"""


# ============================================================
# 主入口
# ============================================================
def classify_and_extract(file_path: str) -> dict:
    """
    文档分类 + OCR + 字段抽取（一步到位）
    :param file_path: 图像/PDF 路径或 URL
    :return: {
        "doc_type": str,
        "sub_template": str | None,
        "confidence": float,
        "fields": dict,
        "transactions": list,
        "notes": str,
        "ocr_text": str,         # 原始 OCR 全文
        "ocr_pages": int,        # OCR 页数
    }
    """
    # ---- Step 1: 调 PaddleOCR-VL 1.6 API ----
    print(f"\n[Step2] 调用 PaddleOCR-VL 1.6 解析: {file_path}")
    ocr_result = get_paddle_ocr().parse(file_path)
    ocr_text = ocr_result["full_text"]
    ocr_pages = len(ocr_result["page_texts"])

    if not ocr_text.strip():
        return {
            "doc_type": "unknown",
            "sub_template": None,
            "confidence": 0.0,
            "fields": {},
            "transactions": [],
            "notes": "OCR 未识别到文本",
            "ocr_text": "",
            "ocr_pages": ocr_pages,
            "quality_flag": "ocr_empty",  # OCR 后校验标记
        }

    # ---- Step 1.5: OCR 后质量门控 ----
    # OCR 字数极少 → OCR 失败（反光/严重模糊/过曝等导致信息丢失）
    # 直接拒绝让用户重拍，不进入后续 LLM 分类流程（等同于 step1 拦截效果）
    ocr_len = len(ocr_text.strip())
    # 信贷材料常见关键词（任一命中即认为可能是材料）
    material_keywords = [
        "身份证", "居民身份证", "护照", "户口", "婚姻",
        "银行", "流水", "账户", "余额", "交易",
        "收入", "证明", "工资", "月薪",
        "营业执照", "企业", "法人", "信用代码",
        "纳税", "税收", "完税",
        "社保", "保险",
        "房产", "不动产", "产权",
        "贷款", "借款", "结清", "申请",
        "许可证", "经营",
        "征信", "信用", "逾期",
        "财务", "报表", "资产", "负债",
        "card", "bank", "license", "certif",
    ]
    has_material_kw = any(kw.lower() in ocr_text.lower() for kw in material_keywords)

    if ocr_len < 10:
        # 区分两种情况：
        # 1. 无材料关键词 → 真正的无关图片（自拍/风景）
        # 2. 有材料关键词但OCR字数极少 → OCR失败（反光/过曝等导致）
        if has_material_kw:
            # OCR 失败：图是材料但信息丢失严重，让用户重拍
            return {
                "doc_type": "unknown",
                "sub_template": None,
                "confidence": 0.0,
                "fields": {},
                "transactions": [],
                "notes": f"OCR 失败（仅 {ocr_len} 字，疑似反光/过曝导致信息丢失），请重拍",
                "ocr_text": ocr_text,
                "ocr_pages": ocr_pages,
                "quality_flag": "ocr_failed",  # OCR 失败，等同于 step1 拦截
            }
        else:
            # 真正的无关图片
            return {
                "doc_type": "unknown",
                "sub_template": None,
                "confidence": 0.0,
                "fields": {},
                "transactions": [],
                "notes": f"疑似无关图片（OCR 仅 {ocr_len} 字且无材料关键词）",
                "ocr_text": ocr_text,
                "ocr_pages": ocr_pages,
                "quality_flag": "irrelevant",  # 无关图片标记
            }

    # ---- Step 2: 硬规则预判（快速识别明显特征，减少 LLM 误判）----
    pre_hint = _hard_rule_preclassify(ocr_text)
    if pre_hint:
        print(f"[Step2] 硬规则预判: {pre_hint}")

    # ---- Step 3: 调千问 LLM 做分类 + 字段抽取（带指数退避重试）----
    print(f"[Step2] 调用千问 LLM 分类+抽取 (OCR {len(ocr_text)} 字)")
    prompt = _build_extract_prompt(ocr_text)
    if pre_hint:
        prompt = f"[硬规则预判提示：可能是 {pre_hint}，请重点验证]\n{prompt}"
    import time as _time
    result = None
    last_err = None
    for _attempt in range(3):  # 最多重试 3 次
        try:
            result = get_qwen().chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.1)
            break
        except json.JSONDecodeError as e:
            # JSON 解析失败，重试一次可能得到不同输出
            print(f"[Step2] LLM 输出 JSON 解析失败 (尝试 {_attempt+1}/3): {e}")
            last_err = e
            _time.sleep(1.5 ** _attempt)  # 1s, 1.5s
        except Exception as e:
            # 限流/超时等，指数退避
            print(f"[Step2] LLM 调用异常 (尝试 {_attempt+1}/3): {str(e)[:120]}")
            last_err = e
            _time.sleep(2 ** _attempt)  # 1s, 2s, 4s
    if result is None:
        result = {
            "doc_type": "unknown",
            "sub_template": None,
            "confidence": 0.0,
            "fields": {},
            "transactions": [],
            "notes": f"LLM 调用失败（已重试3次）: {last_err}",
        }

    # ---- Step 3: 后处理 ----
    # 校验 doc_type 在白名单内
    if result.get("doc_type") not in DOC_TYPES:
        result["doc_type"] = "unknown"

    # 补充原始 OCR 文本
    result["ocr_text"] = ocr_text
    result["ocr_pages"] = ocr_pages

    # ---- 字段兜底：LLM 没抽到的关键字段从 OCR 文本正则提取 ----
    fields = result.get("fields", {})
    import re as _re
    doc_type = result.get("doc_type", "unknown")

    # 姓名兜底（跨文档比对的关键字段）
    # 排除贷款申请表等含多个姓名的复杂文档，避免误提取
    skip_name_types = {"loan_application", "loan_investigation", "financial_statement"}
    if doc_type not in skip_name_types:
        existing_name = fields.get("name") or fields.get("姓名") or fields.get("applicant_name")
        if not existing_name:
            # 只匹配"姓名 XXX"格式，避免误提取
            m = _re.search(r'姓\s*名\s*([^\s,，。；;\<\>\d]{2,4})', ocr_text)
            if m:
                candidate = m.group(1)
                # 严格过滤：必须是中文姓名（2-4个汉字），排除括号、标点、数字、英文字母
                if _re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', candidate):
                    fields["name"] = candidate
                    print(f"[Step2] 姓名兜底提取: {candidate}")

    # 身份证号兜底（跨文档比对的关键字段）
    existing_id = fields.get("id_no") or fields.get("id_number") or fields.get("身份证号") or fields.get("证件号码")
    if not existing_id:
        # 优先匹配"公民身份号码"或"身份证号"后面的18位
        m = _re.search(r'(?:公民身份号码|身份证号|身份证号码|证件号码)[:\s]*(\d{17}[\dXx])', ocr_text)
        if m:
            fields["id_no"] = m.group(1)
            print(f"[Step2] 身份证号兜底提取: {m.group(1)}")
        else:
            # 退化：OCR文本中独立的18位数字（要求前面有"号"等关键词）
            m = _re.search(r'\b\d{17}[\dXx]\b', ocr_text)
            if m and "号" in ocr_text[:ocr_text.find(m.group(0))][-50:]:
                fields["id_no"] = m.group(0)
                print(f"[Step2] 身份证号兜底提取: {m.group(0)}")

    # 流水期间兜底（L3-02 流水月数检查的关键字段）
    if doc_type == "bank_statement":
        existing_period = fields.get("period") or fields.get("起止日期") or fields.get("流水期间") or ""
        if not existing_period:
            # 匹配 "起止日期：20250105-20250625" 或 "流水期间：2025-01至2025-06" 等
            m = _re.search(r'(?:起止日期|流水期间|交易期间|查询期间)[:\s]*([0-9]{4,8}[-/.年]\d{1,2}[-/.月]?日?\s*[至到\-—~]+\s*\d{4,8}[-/.年]?\d{1,2}[-/.月]?日?)', ocr_text)
            if m:
                fields["period"] = m.group(1).strip()
                print(f"[Step2] 流水期间兜底提取: {fields['period']}")

    result["fields"] = fields

    # ---- OCR 后质量标记 ----
    fields = result.get("fields", {})
    fields_filled = sum(1 for v in fields.values() if v is not None and v != "")
    fields_total = len(fields)

    # ---- transactions 后处理：修复 LLM 把 income 抽成 0 的问题 ----
    # LLM 偶发把所有交易 income 填 0.0，但 amount 字段有正值（如 "+8000.00"）
    # 此时从 amount 正值回填 income，避免 L3-06 误触发"流水无任何收入入账"
    if doc_type == "bank_statement":
        txns = result.get("transactions") or []
        has_income = any(
            (t.get("income") is not None and float(t.get("income") or 0) > 0)
            for t in txns
        )
        has_amount_pos = any(
            (t.get("amount") is not None and float(_re.sub(r"[^\d.\-]", "", str(t.get("amount"))) or 0) > 0)
            for t in txns
        )
        if not has_income and has_amount_pos:
            fixed_count = 0
            for t in txns:
                amt_str = str(t.get("amount") or "")
                s_clean = _re.sub(r"[^\d.\-]", "", amt_str)
                try:
                    amt = float(s_clean)
                    if amt > 0 and (t.get("income") is None or float(t.get("income") or 0) == 0):
                        t["income"] = amt
                        fixed_count += 1
                except (ValueError, TypeError):
                    continue
            if fixed_count > 0:
                print(f"[Step2] transactions income 修复: 从 amount 回填 {fixed_count} 笔收入")

    # ---- OCR 内容有效性最终校验 ----
    # LLM 判断 unknown 且置信度极低 → OCR 内容无效（反光/过曝等导致）
    # 标记为 ocr_failed，让 main.py 拦截，等同于 step1 质量门控效果
    if result.get("doc_type") == "unknown" and result.get("confidence", 0) < 0.5:
        result["quality_flag"] = "ocr_failed"
        if not result.get("notes"):
            result["notes"] = "OCR 内容无效（无法识别材料类型），疑似反光/过曝导致，请重拍"
        print(
            f"[Step2] 完成: doc_type=unknown, confidence={result.get('confidence', 0)} "
            f"(ocr_failed - OCR内容无效)"
        )
        return result

    if fields_total > 0:
        fill_rate = fields_filled / fields_total
        if fill_rate < 0.3:
            result["quality_flag"] = "fields_sparse"  # 字段填充率低
        elif fill_rate < 0.6:
            result["quality_flag"] = "fields_partial"  # 字段部分填充
        else:
            result["quality_flag"] = "ok"
    else:
        result["quality_flag"] = "no_schema"  # 该文档类型无字段 schema

    print(
        f"[Step2] 完成: doc_type={result['doc_type']}, "
        f"confidence={result.get('confidence', 0)}, "
        f"fields={fields_filled}/{fields_total} ({result['quality_flag']})"
    )

    return result
