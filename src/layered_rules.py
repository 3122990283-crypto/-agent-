"""
分层规则引擎（layered_rules）

把原来散落在 step3/step4/step5/step6/step11 的检测逻辑统一为四层规则体系：
- L1 材料完整性：必填缺失 / 重复 / 异常材料（原 step5）
- L2 字段逻辑校验：格式校验 + 跨文档一致性（原 step3 + step4）
- L3 业务规则校验：DTI / 逾期 / 收入负债比 / 经营资质 / 流水连续性（原 step6 增强）
- L4 欺诈风险规则：身份证复用 / 团伙 / 设备关联（原 step11 KG 规则）

每条规则统一输出 RuleFinding：
{
    "rule_id": str,           # 规则编号（如 L3-01）
    "layer": "L1"|"L2"|"L3"|"L4",
    "rule_name": str,         # 规则名称
    "severity": "block"|"major"|"minor"|"info",  # 严重程度
    "category": str,          # 问题分类
    "evidence": str,          # 证据描述
    "policy_basis": str,      # 政策依据（可空，由 RAG 补充）
    "docs_involved": list,    # 涉及材料
    "auto_action": str,       # 建议动作: reject/manual_review/warn/info
}

设计要点：
- L1/L2 是硬性校验（材料齐不齐、格式对不对）
- L3 是业务判断（还得起吗、合规吗）
- L4 是反欺诈（骗贷吗）
- severity: block=一票否决, major=重大风险, minor=一般问题, info=提示
- LLM 不参与规则判定，只负责后续解释
"""
import re
from datetime import datetime, date
from typing import Optional

import config
from business_entry import load_strategy


# ============================================================
# RuleFinding 数据结构
# ============================================================
def make_finding(rule_id, layer, rule_name, severity, category,
                 evidence, policy_basis="", docs_involved=None, auto_action="warn") -> dict:
    """构造统一的规则发现"""
    # severity → auto_action 默认映射
    if auto_action == "warn":
        if severity == "block":
            auto_action = "reject"
        elif severity == "major":
            auto_action = "manual_review"
        elif severity == "minor":
            auto_action = "warn"
        elif severity == "info":
            auto_action = "info"
    return {
        "rule_id": rule_id,
        "layer": layer,
        "rule_name": rule_name,
        "severity": severity,
        "category": category,
        "evidence": evidence,
        "policy_basis": policy_basis,
        "docs_involved": docs_involved or [],
        "auto_action": auto_action,
        # 人工审核闭环字段（初始为空，由 review_loop 填充）
        "review_decision": None,      # confirmed / false_positive / escalated
        "reviewer_note": None,
    }


# ============================================================
# L1 材料完整性
# ============================================================
def check_l1_completeness(extracted_docs: list, strategy: dict) -> list:
    """L1 材料完整性检查"""
    findings = []
    submitted_types = set()
    for d in extracted_docs:
        t = d.get("doc_type", "")
        if t and t != "unknown":
            submitted_types.add(t)

    # L1-01: 必填材料缺失
    # severity 从 block 降为 major：缺失允许人工补交，不一票否决
    # 真正的一票否决留给 L2/L4 欺诈类（身份证伪造、一人多证、虚拟货币等）
    required = strategy.get("required_docs", [])
    for req in required:
        if req.get("required", True) and req["doc_type"] not in submitted_types:
            findings.append(make_finding(
                "L1-01", "L1", "必填材料缺失", "major", "材料完整性",
                f"缺失必填材料：{req['label']}（{req['doc_type']}），需补交",
                docs_involved=[],
                auto_action="manual_review",
            ))

    # L1-02: 重复材料
    type_count = {}
    for d in extracted_docs:
        t = d.get("doc_type", "")
        if t and t != "unknown":
            type_count[t] = type_count.get(t, 0) + 1
    for t, n in type_count.items():
        if n > 1 and t not in ("bank_statement",):  # 流水允许多张
            findings.append(make_finding(
                "L1-02", "L1", "重复材料", "minor", "材料完整性",
                f"材料类型 {t} 提交了 {n} 份，存在重复",
                docs_involved=[t],
            ))

    # L1-03: OCR 失败材料
    for d in extracted_docs:
        flag = d.get("quality_flag", "")
        if flag in ("ocr_failed", "ocr_empty"):
            findings.append(make_finding(
                "L1-03", "L1", "OCR失败材料", "major", "材料完整性",
                f"材料 {d.get('file','')} OCR 识别失败（{flag}），信息无法提取",
                docs_involved=[d.get("doc_type", "unknown")],
                auto_action="manual_review",
            ))

    return findings


# ============================================================
# L2 字段逻辑校验
# ============================================================
def check_l2_field_logic(extracted_docs: list, strategy: dict) -> list:
    """L2 字段格式 + 跨文档一致性"""
    findings = []

    # ---- L2-01: 身份证号格式校验 ----
    id_numbers = {}  # {id_num: [doc_type,...]}
    for d in extracted_docs:
        f = d.get("fields", {})
        idn = f.get("id_number") or f.get("身份证号") or ""
        if idn:
            idn = str(idn).strip()
            # 脱敏身份证号（含 *）跳过校验，不报错也不参与一致性比对
            if "*" in idn:
                continue
            # 格式校验
            if not re.match(r"^\d{17}[\dXx]$", idn):
                findings.append(make_finding(
                    "L2-01", "L2", "身份证号格式错误", "major", "字段逻辑",
                    f"身份证号 {idn} 格式不正确（应18位）",
                    docs_involved=[d.get("doc_type")],
                ))
            else:
                # 校验位验证
                if not _verify_id_checksum(idn):
                    findings.append(make_finding(
                        "L2-01", "L2", "身份证号校验位错误", "block", "字段逻辑",
                        f"身份证号 {idn} 校验位不正确，疑似伪造",
                        docs_involved=[d.get("doc_type")],
                        auto_action="reject",
                    ))
                id_numbers.setdefault(idn, []).append(d.get("doc_type"))

    # ---- L2-02: 跨文档身份证号不一致 ----
    if len(id_numbers) > 1:
        findings.append(make_finding(
            "L2-02", "L2", "身份证号跨文档不一致", "block", "跨文档一致性",
            f"申请人出现 {len(id_numbers)} 个不同身份证号：{list(id_numbers.keys())}，身份主体混乱",
            docs_involved=sum(id_numbers.values(), []),
            auto_action="reject",
        ))

    # ---- L2-03: 姓名跨文档不一致 ----
    # 排除结婚证/离婚证：其 name 字段可能是配偶姓名，非主申请人
    # 排除银行流水：其户名（account_holder）可能是企业名（经营贷），不应参与个人姓名比对
    #   bank_statement 的 schema 字段是 account_holder 而非 name；LLM 偶发误填 name 字段时会引入企业名噪声
    # 姓名相似度容错：编辑距离 ≤1 视为同一人（OCR 漏字/形近字）
    _SKIP_NAME_TYPES = {"marriage_cert", "divorce_cert", "bank_statement"}
    names = {}
    for d in extracted_docs:
        if d.get("doc_type") in _SKIP_NAME_TYPES:
            continue
        f = d.get("fields", {})
        name = f.get("name") or f.get("姓名") or ""
        if name:
            name = str(name).strip()
            # 必须是中文且 ≥2 字
            if not re.search(r"[\u4e00-\u9fa5]{2,}", name):
                continue
            names.setdefault(name, []).append(d.get("doc_type"))

    def _edit_distance(s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    # 合并相似姓名（编辑距离 ≤1 视为同一人，OCR 容错）
    name_list = list(names.keys())
    merged = {}  # {代表名: [原名1, 原名2, ...]}
    used = set()
    for i, n1 in enumerate(name_list):
        if n1 in used:
            continue
        merged[n1] = [n1]
        used.add(n1)
        for j in range(i + 1, len(name_list)):
            n2 = name_list[j]
            if n2 in used:
                continue
            # 长度差 ≤1 且编辑距离 ≤1 视为同一人（OCR 容错）
            if abs(len(n1) - len(n2)) <= 1 and _edit_distance(n1, n2) <= 1:
                # 中文姓名特殊处理：若姓氏不同，不合并（"张伟" vs "李伟" 是不同人）
                # 仅在编辑距离=0（完全相同）或姓氏相同时才合并
                if n1 == n2:
                    merged[n1].append(n2)
                    used.add(n2)
                elif n1[0] == n2[0]:
                    # 同姓，可能是 OCR 漏字/形近字，合并
                    merged[n1].append(n2)
                    used.add(n2)
                # 不同姓的编辑距离=1 不合并（如"张伟"vs"李伟"）

    # 合并后的姓名组数 >1 才是真实不一致
    if len(merged) > 1:
        # 重建显示用 dict
        display = {}
        for rep, alts in merged.items():
            docs = []
            for alt in alts:
                docs.extend(names[alt])
            display[rep] = docs
        findings.append(make_finding(
            "L2-03", "L2", "姓名跨文档不一致", "block", "跨文档一致性",
            f"申请人姓名在材料间不一致：{display}",
            docs_involved=sum(display.values(), []),
            auto_action="reject",
        ))

    # ---- L2-04: 收入跨文档差异 ----
    incomes = {}
    for d in extracted_docs:
        f = d.get("fields", {})
        inc = f.get("monthly_income") or f.get("月收入") or f.get("income") or ""
        if inc:
            try:
                inc_val = float(re.sub(r"[^\d.]", "", str(inc)))
                if inc_val > 0:
                    incomes.setdefault(d.get("doc_type"), inc_val)
            except (ValueError, TypeError):
                pass
    if len(incomes) >= 2:
        vals = list(incomes.values())
        diff = abs(max(vals) - min(vals)) / max(vals)
        threshold = strategy.get("thresholds", {}).get("income_mismatch", 0.2)
        if diff > threshold:
            findings.append(make_finding(
                "L2-04", "L2", "收入跨文档差异过大", "major", "跨文档一致性",
                f"收入数据跨文档差异 {diff*100:.1f}%（阈值 {threshold*100:.0f}%）：{incomes}",
                docs_involved=list(incomes.keys()),
                auto_action="manual_review",
            ))

    # ---- L2-05: 身份证有效期过期 ----
    # 字段别名：id_card_front/back 的 schema 字段名是 valid_until（英文），
    # 也兼容 expiry_date / 有效期至 等历史命名
    today = date.today()
    for d in extracted_docs:
        if d.get("doc_type") in ("id_card_front", "id_card_back"):
            f = d.get("fields", {})
            exp = (f.get("valid_until") or f.get("expiry_date")
                   or f.get("有效期至") or f.get("有效期") or "")
            if exp:
                # "长期" 视为有效，跳过
                if str(exp).strip() in ("长期", "长期有效", "long"):
                    continue
                exp_date = _parse_date(exp)
                if exp_date and exp_date < today:
                    findings.append(make_finding(
                        "L2-05", "L2", "身份证已过期", "block", "字段逻辑",
                        f"身份证有效期至 {exp}，已过期",
                        docs_involved=[d.get("doc_type")],
                        auto_action="reject",
                    ))

    # ---- L2-06: 经营贷流水户名与营业执照企业名不一致 ----
    # 经营贷场景下，银行流水户名应为借款企业，应与营业执照企业名一致
    # 若不一致 → 账户主体疑似非借款企业，存在欺诈风险
    def _normalize_company(name: str) -> str:
        """企业名归一化：去除常见后缀、首尾空格、全角字符"""
        if not name:
            return ""
        s = str(name).strip()
        # 去除常见公司后缀（注意顺序：长的在前，避免误匹配）
        for suffix in ["有限责任公司", "股份有限公司", "有限公司", "股份公司", "合伙企业"]:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
                break
        return s.strip()

    if strategy.get("category") == "corporate":
        bs_holder = None
        bl_company = None
        for d in extracted_docs:
            dt = d.get("doc_type", "")
            f = d.get("fields", {})
            if dt == "bank_statement":
                # 兼容多种字段命名：account_holder（schema）/ account_name（测试数据）/ 户名 / name（LLM 偶发）
                bs_holder = (f.get("account_holder") or f.get("account_name")
                             or f.get("户名") or f.get("name") or bs_holder)
            elif dt == "business_license":
                bl_company = (f.get("company_name") or f.get("企业名称")
                              or f.get("名称") or f.get("name") or bl_company)
        if bs_holder and bl_company:
            bs_s = str(bs_holder).strip()
            bl_s = str(bl_company).strip()
            # 归一化后比较，避免"XX有限公司"vs"XX有限责任公司"误报
            bs_norm = _normalize_company(bs_s)
            bl_norm = _normalize_company(bl_s)
            if bs_norm and bl_norm and bs_norm != bl_norm:
                findings.append(make_finding(
                    "L2-06", "L2", "流水户名与营业执照企业名不一致", "block", "跨文档一致性",
                    f"银行流水户名 '{bs_s}' 与营业执照企业名 '{bl_s}' 不一致，账户主体非借款企业",
                    docs_involved=["bank_statement", "business_license"],
                    auto_action="reject",
                ))

    # ---- L2-07: 申请表与调查报告关键字段矛盾 ----
    # 贷款金额/期限/职位 等关键字段在申请表和调查报告间应保持一致
    app_fields = {}
    inv_fields = {}
    for d in extracted_docs:
        dt = d.get("doc_type", "")
        if dt == "loan_application":
            app_fields = d.get("fields", {})
        elif dt == "loan_investigation":
            inv_fields = d.get("fields", {})
    if app_fields and inv_fields:
        contradictions = []
        # 比对字段：扩展字段别名，兼容测试数据中"申请金额"/"申请期限"等命名
        field_aliases = [
            ("loan_amount", ["贷款金额", "申请金额", "applied_amount", "loan_amount_value"]),
            ("loan_term", ["贷款期限", "申请期限", "applied_term", "loan_term_value"]),
            ("position", ["职位", "职务", "工作岗位", "position_value"]),
            ("annual_income", ["年收入", "年收入金额", "annual_income_value", "月收入"]),
        ]
        for fkey, aliases in field_aliases:
            # 在两个文档中查找字段值（先 schema key 再别名）
            v1 = app_fields.get(fkey)
            if v1 is None:
                for alias in aliases:
                    if app_fields.get(alias) is not None:
                        v1 = app_fields.get(alias)
                        break
            v2 = inv_fields.get(fkey)
            if v2 is None:
                for alias in aliases:
                    if inv_fields.get(alias) is not None:
                        v2 = inv_fields.get(alias)
                        break
            if v1 is None or v2 is None:
                continue
            s1, s2 = str(v1).strip(), str(v2).strip()
            if not s1 or not s2:
                continue
            if s1 == s2:
                continue
            # 数值类字段：差异 >20% 才报，避免小额舍入误差
            n1 = re.sub(r"[^\d.]", "", s1)
            n2 = re.sub(r"[^\d.]", "", s2)
            try:
                f1, f2 = float(n1), float(n2)
                if f1 > 0 and f2 > 0:
                    diff = abs(f1 - f2) / max(f1, f2)
                    if diff <= 0.2:
                        continue  # 差异 ≤20% 容忍
                    contradictions.append(f"{aliases[0]}: 申请表={s1} vs 调查报告={s2}（差异 {diff*100:.0f}%）")
                    continue
            except (ValueError, ZeroDivisionError):
                pass
            # 非数值字段：直接报
            contradictions.append(f"{aliases[0]}: 申请表={s1} vs 调查报告={s2}")
        if contradictions:
            findings.append(make_finding(
                "L2-07", "L2", "申请表与调查报告字段矛盾", "major", "跨文档一致性",
                "；".join(contradictions),
                docs_involved=["loan_application", "loan_investigation"],
                auto_action="manual_review",
            ))

    # ---- L2-08: 流水期间与交易明细日期矛盾 ----
    # 流水 period 标注的起止月 vs transactions 实际日期所在月份应一致
    # 不一致 → 疑似时间伪造（如 period 标 2023 但交易日期全在 2025）
    for d in extracted_docs:
        if d.get("doc_type") != "bank_statement":
            continue
        f = d.get("fields", {})
        period_str = f.get("period") or f.get("起止日期") or f.get("流水期间") or ""
        # 强制转换为字符串，避免 period 是数字/None 导致 re.search 报错
        period_str = str(period_str) if period_str is not None else ""
        txns = d.get("transactions") or []
        if not period_str or not txns:
            continue
        # 提取 period 的年月范围
        m_start = re.search(r"(\d{4})[-/.年]?(\d{1,2})", period_str)
        m_end = re.search(r"(\d{4})[-/.年]?(\d{1,2})(?:[^\d]*$)", period_str)
        if not m_start:
            continue
        try:
            y1, mo1 = int(m_start.group(1)), int(m_start.group(2))
        except (ValueError, IndexError):
            continue
        # 提取交易明细的最早年月
        txn_ym = set()
        for t in txns:
            ds = str(t.get("date", ""))
            m = re.match(r"(\d{4})[-/]?\d{1,2}", ds)
            if m:
                txn_ym.add(int(m.group(1)))
        if not txn_ym:
            continue
        period_year = y1
        txn_years = txn_ym
        # 检查交易年份是否在 period 年份附近（±1年内算正常，>1年差视为矛盾）
        # 简化判断：period 起始年 vs 交易最早年，差异 >1 年视为矛盾
        if abs(min(txn_years) - period_year) > 1:
            findings.append(make_finding(
                "L2-08", "L2", "流水期间与交易日期矛盾", "major", "跨文档一致性",
                f"流水 period 起始年 {period_year}，但交易明细日期所在年为 {sorted(txn_years)}，疑似时间伪造",
                docs_involved=["bank_statement"],
                auto_action="manual_review",
            ))
            break

    return findings


def _verify_id_checksum(id_num: str) -> bool:
    """身份证校验位验证"""
    if len(id_num) != 18:
        return False
    weights = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
    codes = "10X98765432"
    try:
        total = sum(int(id_num[i]) * weights[i] for i in range(17))
        return codes[total % 11] == id_num[17].upper()
    except (ValueError, IndexError):
        return False


def _parse_date(s: str) -> Optional[date]:
    """解析多种日期格式"""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y年%m月%d日", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _calc_period_months(period_str: str) -> int:
    """从流水期间字符串计算覆盖月数
    支持格式：
      - "20250105-20250625" (YYYYMMDD-YYYYMMDD)
      - "2025-01-05-2025-06-25"
      - "2025/01/05-2025/06/25"
      - "2025年01月05日至2025年06月25日"
      - "2025-01 至 2025-06" (YYYY-MM)
    返回覆盖的月数（含起止月），解析失败返回 0
    """
    if not period_str:
        return 0
    s = str(period_str).strip().replace("/", "-")
    # 提取两组 日期/年月
    # 先尝试匹配 YYYYMMDD-YYYYMMDD（8位纯数字）
    m = re.match(r"(\d{8})[-至到—]+(\d{8})", s)
    if m:
        d1 = _parse_date(m.group(1))
        d2 = _parse_date(m.group(2))
        if d1 and d2:
            return (d2.year - d1.year) * 12 + (d2.month - d1.month) + 1
    # 匹配 YYYY-MM-DD ... YYYY-MM-DD（含分隔符）
    m = re.search(r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\s*[-至到—~]+\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})", s)
    if m:
        d1 = _parse_date(m.group(1))
        d2 = _parse_date(m.group(2))
        if d1 and d2:
            return (d2.year - d1.year) * 12 + (d2.month - d1.month) + 1
    # 匹配 YYYY年MM月DD日 ... YYYY年MM月DD日
    m = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)\s*[至到\-—~]+\s*(\d{4}年\d{1,2}月\d{1,2}日)", s)
    if m:
        d1 = _parse_date(m.group(1))
        d2 = _parse_date(m.group(2))
        if d1 and d2:
            return (d2.year - d1.year) * 12 + (d2.month - d1.month) + 1
    # 匹配 YYYY-MM ... YYYY-MM（仅年月）
    m = re.search(r"(\d{4}[-/.]\d{1,2})\s*[至到\-—~]+\s*(\d{4}[-/.]\d{1,2})", s)
    if m:
        try:
            y1, mo1 = re.split(r"[-/.]", m.group(1))
            y2, mo2 = re.split(r"[-/.]", m.group(2))
            return (int(y2) - int(y1)) * 12 + (int(mo2) - int(mo1)) + 1
        except (ValueError, IndexError):
            pass
    return 0


# ============================================================
# L3 业务规则校验
# ============================================================
def check_l3_business_rules(extracted_docs: list, strategy: dict, rule_result: dict = None) -> list:
    """L3 业务规则：DTI / 逾期 / 流水连续性 / 经营资质"""
    findings = []
    thresholds = strategy.get("thresholds", {})

    # ---- L3-01: 征信报告逾期次数 ----
    for d in extracted_docs:
        if d.get("doc_type") == "credit_report":
            f = d.get("fields", {})
            overdue_total = f.get("overdue_total") or f.get("累计逾期") or 0
            overdue_consec = f.get("overdue_consecutive") or f.get("连续逾期") or 0
            try:
                ot = int(overdue_total)
                oc = int(overdue_consec)
            except (ValueError, TypeError):
                continue
            ot_thr = thresholds.get("overdue_total_threshold", 6)
            oc_thr = thresholds.get("overdue_consecutive_threshold", 3)
            if ot >= ot_thr:
                findings.append(make_finding(
                    "L3-01", "L3", "征信累计逾期超标", "block", "业务规则",
                    f"近2年累计逾期 {ot} 次（阈值 {ot_thr}）",
                    docs_involved=["credit_report"],
                    auto_action="reject",
                ))
            if oc >= oc_thr:
                findings.append(make_finding(
                    "L3-01", "L3", "征信连续逾期超标", "block", "业务规则",
                    f"近2年连续逾期 {oc} 次（阈值 {oc_thr}）",
                    docs_involved=["credit_report"],
                    auto_action="reject",
                ))

    # ---- L3-02: 流水月数不足 ----
    # 优先用 transactions 实际日期计算覆盖月数；
    # transactions 不足时用 period 字段兜底，但同时标记"交易明细月数不足"为 minor
    # 避免 period 字段被伪造（如标 18 个月但交易只有 2 个月）导致漏判
    flow_month_min = thresholds.get("flow_month_min", 6)
    for d in extracted_docs:
        if d.get("doc_type") == "bank_statement":
            txns = d.get("transactions") or []
            months = set()
            # 方式1：从 transactions 日期提取月份（实际数据，可信度高）
            for t in txns:
                d_str = t.get("date") or ""
                # 兼容多种日期格式：2025-01-05 / 20250105 / 2025/01/05
                # 用 search 而非 match，避免前导空格导致漏匹配
                # 统一归一化为 YYYY-MM 格式，避免 "202501" 与 "2025-01" 被算作不同月份
                m = re.search(r"(\d{4})[-/]?(\d{2})", d_str)
                if m:
                    months.add(f"{m.group(1)}-{m.group(2)}")
            txn_month_count = len(months)
            # 方式2：transactions 不足时，从 period/起止日期 字段计算月数（兜底，可信度低）
            if txn_month_count < flow_month_min:
                f = d.get("fields", {})
                period_str = f.get("period") or f.get("起止日期") or f.get("流水期间") or ""
                period_months = _calc_period_months(period_str)
                if period_months > txn_month_count:
                    # period 标注月数 > 实际交易月数：可能 period 被伪造或交易明细缺失
                    # 取 period 月数作为最终月数，但同时报 minor 提示交易明细不足
                    months = set(range(period_months))
                    if txn_month_count > 0 and period_months >= flow_month_min:
                        # period 满足要求但交易明细不足 → 标记为 minor 提示
                        findings.append(make_finding(
                            "L3-02", "L3", "流水交易明细月数不足", "minor", "业务规则",
                            f"流水 period 标注 {period_months} 个月，但交易明细仅覆盖 {txn_month_count} 个月，"
                            f"建议核实 period 标注真实性",
                            docs_involved=["bank_statement"],
                            auto_action="warn",
                        ))
            if months and len(months) < flow_month_min:
                findings.append(make_finding(
                    "L3-02", "L3", "流水月数不足", "major", "业务规则",
                    f"流水仅覆盖 {len(months)} 个月（要求 ≥{flow_month_min} 个月）",
                    docs_involved=["bank_statement"],
                    auto_action="manual_review",
                ))

    # ---- L3-03: 收入负债比 (DTI) ----
    # 复用 step6 计算的 avg_monthly_income
    if rule_result and rule_result.get("avg_monthly_income"):
        avg_income = rule_result["avg_monthly_income"]
        # 估算月负债（简化：从征信报告取）
        monthly_debt = 0
        for d in extracted_docs:
            if d.get("doc_type") == "credit_report":
                f = d.get("fields", {})
                try:
                    monthly_debt = float(f.get("monthly_debt") or f.get("月负债") or 0)
                except (ValueError, TypeError):
                    monthly_debt = 0
        if avg_income > 0 and monthly_debt > 0:
            dti = monthly_debt / avg_income
            dti_reject = thresholds.get("dti_reject", 0.7)
            dti_warn = thresholds.get("dti_warning", 0.5)
            if dti >= dti_reject:
                findings.append(make_finding(
                    "L3-03", "L3", "收入负债比超标", "block", "业务规则",
                    f"DTI={dti*100:.1f}%（拒贷阈值 {dti_reject*100:.0f}%），偿债能力不足",
                    docs_involved=["bank_statement", "credit_report"],
                    auto_action="reject",
                ))
            elif dti >= dti_warn:
                findings.append(make_finding(
                    "L3-03", "L3", "收入负债比偏高", "minor", "业务规则",
                    f"DTI={dti*100:.1f}%（预警阈值 {dti_warn*100:.0f}%）",
                    docs_involved=["bank_statement", "credit_report"],
                ))

    # ---- L3-04: 经营资质过期（经营贷专用）----
    if strategy.get("category") == "corporate":
        today = date.today()
        for d in extracted_docs:
            if d.get("doc_type") in ("business_license", "food_license"):
                f = d.get("fields", {})
                exp = f.get("expiry_date") or f.get("有效期至") or ""
                if exp:
                    exp_date = _parse_date(exp)
                    if exp_date and exp_date < today:
                        findings.append(make_finding(
                            "L3-04", "L3", "经营资质过期", "block", "业务规则",
                            f"{d.get('doc_type')} 有效期至 {exp}，已过期",
                            docs_involved=[d.get("doc_type")],
                            auto_action="reject",
                        ))

    # ---- L3-05: 流水币种异常 ----
    suspicious_currency = ["USDT", "稳定币", "虚拟币", "BTC", "ETH", "泰达币"]
    for d in extracted_docs:
        if d.get("doc_type") == "bank_statement":
            ocr_text = d.get("ocr_text", "") or ""
            for cur in suspicious_currency:
                if cur in ocr_text.upper() or cur in ocr_text:
                    findings.append(make_finding(
                        "L3-05", "L3", "流水币种异常", "block", "业务规则",
                        f"流水中出现可疑币种：{cur}，疑似虚拟货币交易",
                        docs_involved=["bank_statement"],
                        auto_action="reject",
                    ))
                    break

    # ---- L3-06: 收入证明与流水实际入账严重不符 ----
    # 收入证明年化收入 vs 银行流水正向入账总额，差异过大说明收入虚报或流水造假
    # 经营贷分支：无 income_certificate，用 fund_flow_analysis 的月均入账×12 代替
    income_cert_annual = None
    income_cert_source = None  # 记录来源：income_certificate / fund_flow_analysis / loan_investigation
    for d in extracted_docs:
        if d.get("doc_type") == "income_certificate":
            f = d.get("fields", {})
            # 优先年收入，月收入兜底（×12 年化）
            # 注意：用 is not None 判断，避免 0 被 falsy 误判
            inc = f.get("annual_income")
            if inc is None:
                inc = f.get("年收入")
            if inc is not None:
                try:
                    v = float(re.sub(r"[^\d.]", "", str(inc)))
                    if v > 0:
                        income_cert_annual = v
                        income_cert_source = "income_certificate"
                except (ValueError, TypeError):
                    pass
            if income_cert_annual is None:
                inc_m = f.get("monthly_income")
                if inc_m is None:
                    inc_m = f.get("月收入")
                if inc_m is None:
                    inc_m = f.get("income")
                if inc_m is not None:
                    try:
                        v = float(re.sub(r"[^\d.]", "", str(inc_m)))
                        if v > 0:
                            income_cert_annual = v * 12
                            income_cert_source = "income_certificate"
                    except (ValueError, TypeError):
                        pass
            break

    # 经营贷分支：无收入证明时，用资金流水分析表的月均流入×12 作为申报收入
    if income_cert_annual is None:
        for d in extracted_docs:
            if d.get("doc_type") == "fund_flow_analysis":
                f = d.get("fields", {})
                # 兼容多种字段命名：monthly_avg_inflow（schema）/ monthly_avg_inflow_value（模板）/ 月均流入
                m_avg = (f.get("monthly_avg_inflow") or f.get("monthly_avg_inflow_value")
                         or f.get("月均流入") or f.get("月均流入金额") or "")
                if m_avg:
                    try:
                        v = float(re.sub(r"[^\d.]", "", str(m_avg)))
                        if v > 0:
                            income_cert_annual = v * 12
                            income_cert_source = "fund_flow_analysis"
                    except (ValueError, TypeError):
                        pass
                break
    # 经营贷分支 2：贷款调查报告的年收入
    if income_cert_annual is None:
        for d in extracted_docs:
            if d.get("doc_type") == "loan_investigation":
                f = d.get("fields", {})
                inc = f.get("annual_income") or f.get("年收入") or ""
                if inc:
                    try:
                        v = float(re.sub(r"[^\d.]", "", str(inc)))
                        if v > 0:
                            income_cert_annual = v
                            income_cert_source = "loan_investigation"
                    except (ValueError, TypeError):
                        pass
                break

    # 收入证明抽取异常检测：
    # income_certificate 来源的年化值超过 100 万元/年（约 8.3 万/月）时，
    # 大概率是 LLM 抽取错误（如多读一位数字、小数点错位），而非真实收入虚报。
    # 此时 L3-06 的 block/major 应降级为 minor，避免抽取错误导致误判。
    # 注意：fund_flow_analysis（经营贷经营流水）和 loan_investigation 不受此限制，
    # 因为企业经营流水和调查报告核实过的年收入可能真实较高。
    _INCOME_CERT_SUSPECTED_ERROR = (
        income_cert_source == "income_certificate"
        and income_cert_annual is not None
        and income_cert_annual > 1_000_000  # 100 万/年
    )

    # 来源标签：用于 evidence 中准确描述年化收入来源，避免经营贷场景误导
    _SOURCE_LABELS = {
        "income_certificate": "收入证明年化",
        "fund_flow_analysis": "资金流水分析月均流入年化",
        "loan_investigation": "调查报告年收入",
    }
    _income_source_label = _SOURCE_LABELS.get(income_cert_source, "申报年收入")
    _income_source_doc = income_cert_source or "income_certificate"

    flow_annual_income = 0.0
    flow_has_income = False
    flow_has_amount_field = False  # transactions 是否有 amount/income 字段
    flow_txn_count = 0
    for d in extracted_docs:
        if d.get("doc_type") == "bank_statement":
            txns = d.get("transactions") or []
            flow_txn_count = len(txns)
            for t in txns:
                # 优先读 income 字段（schema 标准），其次读 amount（含正负号）
                # 注意：用 `in` 判断字段存在，避免 0.0 被 falsy 误判
                if "income" in t and t.get("income") is not None:
                    flow_has_amount_field = True
                    try:
                        amt = float(re.sub(r"[^\d.]", "", str(t.get("income"))))
                        if amt > 0:
                            flow_annual_income += amt
                            flow_has_income = True
                            continue
                    except (ValueError, TypeError):
                        pass
                # 兼容 amount 字段（正数=收入，负数=支出）
                if "amount" in t and t.get("amount") is not None:
                    flow_has_amount_field = True
                    try:
                        s = str(t.get("amount")).strip()
                        s_clean = re.sub(r"[^\d.\-]", "", s)
                        amt = float(s_clean)
                        if amt > 0:
                            flow_annual_income += amt
                            flow_has_income = True
                    except (ValueError, TypeError):
                        pass

    if income_cert_annual and income_cert_annual > 0 and flow_has_income and flow_annual_income > 0:
        diff_ratio = abs(income_cert_annual - flow_annual_income) / max(income_cert_annual, flow_annual_income)
        if diff_ratio > 0.5:
            # 收入证明抽取异常（年化>100万）时，差异大概率是抽取错误而非真实虚报，降级 minor
            if _INCOME_CERT_SUSPECTED_ERROR:
                findings.append(make_finding(
                    "L3-06", "L3", "收入证明抽取值异常", "minor", "业务规则",
                    f"{_income_source_label} {income_cert_annual:.0f}元（超过 100 万，疑似抽取错误），"
                    f"流水正向入账 {flow_annual_income:.0f}元（差异 {diff_ratio*100:.0f}%），"
                    f"建议人工核实收入证明实际金额",
                    docs_involved=[_income_source_doc, "bank_statement"],
                    auto_action="warn",
                ))
            else:
                findings.append(make_finding(
                    "L3-06", "L3", "收入证明与流水严重不符", "block", "业务规则",
                    f"{_income_source_label} {income_cert_annual:.0f}元 vs 流水正向入账 {flow_annual_income:.0f}元"
                    f"（差异 {diff_ratio*100:.0f}%），收入严重虚高或流水造假",
                    docs_involved=[_income_source_doc, "bank_statement"],
                    auto_action="reject",
                ))
        elif diff_ratio > 0.2:
            # 差异 20%-50%：抽取异常时降级 minor，否则保持 major
            if _INCOME_CERT_SUSPECTED_ERROR:
                findings.append(make_finding(
                    "L3-06", "L3", "收入证明抽取值异常", "minor", "业务规则",
                    f"{_income_source_label} {income_cert_annual:.0f}元（超过 100 万，疑似抽取错误），"
                    f"流水正向入账 {flow_annual_income:.0f}元（差异 {diff_ratio*100:.0f}%），"
                    f"建议人工核实",
                    docs_involved=[_income_source_doc, "bank_statement"],
                    auto_action="warn",
                ))
            else:
                findings.append(make_finding(
                    "L3-06", "L3", "收入证明与流水差异较大", "major", "业务规则",
                    f"{_income_source_label} {income_cert_annual:.0f}元 vs 流水正向入账 {flow_annual_income:.0f}元"
                    f"（差异 {diff_ratio*100:.0f}%）",
                    docs_involved=[_income_source_doc, "bank_statement"],
                    auto_action="manual_review",
                ))
    elif (income_cert_annual and income_cert_annual > 0
          and flow_has_amount_field and not flow_has_income and flow_txn_count > 0):
        # 收入证明有金额 + 流水有交易明细且字段完整但完全无正向入账
        # 注意：若 transactions 无 amount/income 字段（字段抽取失败），不触发，避免误判
        # 额外检查：OCR 文本中是否有收入关键词（工资/薪金/代发/入账等）
        # 若有收入关键词但 income=0，可能是 LLM 抽取问题而非真实无收入，降级为 minor
        _INCOME_KEYWORDS = ("工资", "薪金", "代发", "代发工资", "入账", "存入",
                            "汇入", "转入", "收到", "销售款", "货款")
        bs_ocr_text = ""
        for d in extracted_docs:
            if d.get("doc_type") == "bank_statement":
                bs_ocr_text = d.get("ocr_text", "") or ""
                break
        has_income_keyword = any(kw in bs_ocr_text for kw in _INCOME_KEYWORDS)
        if has_income_keyword or _INCOME_CERT_SUSPECTED_ERROR:
            # OCR 文本含收入关键词 或 收入证明抽取异常 → 可能是 LLM 抽取问题，降级为 minor
            reason = "OCR 文本含收入关键词" if has_income_keyword else "收入证明年化值异常（>100万）"
            findings.append(make_finding(
                "L3-06", "L3", "流水收入字段抽取异常", "minor", "业务规则",
                f"{_income_source_label} {income_cert_annual:.0f}元，流水 {flow_txn_count} 条交易 income 全为 0，"
                f"但 {reason}，疑似 LLM 抽取问题，建议人工核实流水实际入账",
                docs_involved=[_income_source_doc, "bank_statement"],
                auto_action="warn",
            ))
        else:
            # OCR 文本也无收入关键词 → 真实无收入入账
            findings.append(make_finding(
                "L3-06", "L3", "流水无任何收入入账", "block", "业务规则",
                f"{_income_source_label} {income_cert_annual:.0f}元，但银行流水 {flow_txn_count} 条交易无任何正向入账，严重不符",
                docs_involved=[_income_source_doc, "bank_statement"],
                auto_action="reject",
            ))

    return findings


# ============================================================
# L4 欺诈风险规则
# ============================================================
def check_l4_fraud(extracted_docs: list, kg_alerts: list = None, strategy: dict = None) -> list:
    """L4 欺诈风险：身份证复用 / 团伙 / 设备关联"""
    findings = []

    # ---- L4-01: KG 反欺诈告警（来自 step11）----
    if kg_alerts:
        for alert in kg_alerts:
            findings.append(make_finding(
                "L4-01", "L4", "知识图谱反欺诈告警", "block", "欺诈风险",
                alert.get("evidence", ""),
                policy_basis=alert.get("policy_basis", ""),
                docs_involved=alert.get("applicants", []),
                auto_action="reject",
            ))

    # ---- L4-02: 多身份证号（单申请人，R8 等价检测）----
    id_set = set()
    for d in extracted_docs:
        f = d.get("fields", {})
        idn = f.get("id_number") or f.get("身份证号") or ""
        if idn and re.match(r"^\d{17}[\dXx]$", str(idn).strip()):
            id_set.add(str(idn).strip().upper())
    if len(id_set) > 1:
        findings.append(make_finding(
            "L4-02", "L4", "一人多身份证号", "block", "欺诈风险",
            f"申请人在不同材料中使用 {len(id_set)} 个身份证号：{list(id_set)}",
            policy_basis="《信贷审批负面客户认定标准》第十条 关键身份标识无法交叉验证",
            auto_action="reject",
        ))

    # ---- L4-03: 配偶证件号冲突 ----
    spouse_ids = set()
    for d in extracted_docs:
        if d.get("doc_type") in ("marriage_cert", "loan_application"):
            f = d.get("fields", {})
            sid = f.get("spouse_id_number") or f.get("配偶证件号") or ""
            if sid and re.match(r"^\d{17}[\dXx]$", str(sid).strip()):
                spouse_ids.add(str(sid).strip().upper())
    if len(spouse_ids) > 1:
        findings.append(make_finding(
            "L4-03", "L4", "配偶证件号冲突", "major", "欺诈风险",
            f"配偶证件号在材料间不一致：{list(spouse_ids)}",
            auto_action="manual_review",
        ))

    return findings


# ============================================================
# 统一入口
# ============================================================
def run_all_layers(extracted_docs: list, strategy: dict,
                   rule_result: dict = None, kg_alerts: list = None) -> dict:
    """
    运行四层规则引擎
    :return: {
        "findings": [RuleFinding, ...],
        "layer_summary": {L1: count, L2: count, ...},
        "severity_summary": {block: n, major: n, minor: n, info: n},
        "auto_action": "reject"|"manual_review"|"warn"|"pass",  # 综合建议
        "block_count": int,
    }
    """
    findings = []
    findings += check_l1_completeness(extracted_docs, strategy)
    findings += check_l2_field_logic(extracted_docs, strategy)
    findings += check_l3_business_rules(extracted_docs, strategy, rule_result)
    findings += check_l4_fraud(extracted_docs, kg_alerts, strategy)

    # 分层统计
    layer_summary = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    severity_summary = {"block": 0, "major": 0, "minor": 0, "info": 0}
    for f in findings:
        layer_summary[f["layer"]] = layer_summary.get(f["layer"], 0) + 1
        severity_summary[f["severity"]] = severity_summary.get(f["severity"], 0) + 1

    # 综合动作建议
    block_count = severity_summary["block"]
    major_count = severity_summary["major"]
    if block_count > 0:
        auto_action = "reject"
    elif major_count > 0:
        auto_action = "manual_review"
    elif severity_summary["minor"] > 0:
        auto_action = "warn"
    else:
        auto_action = "pass"

    return {
        "findings": findings,
        "layer_summary": layer_summary,
        "severity_summary": severity_summary,
        "auto_action": auto_action,
        "block_count": block_count,
        "major_count": major_count,
    }


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    from business_entry import detect_product

    # 模拟 12-谢衍 的材料（一人多身份证号 + 配偶证件冲突）
    docs = [
        {"doc_type": "business_license", "fields": {"expiry_date": "2009-12-31"}, "file": "license.jpg"},
        {"doc_type": "id_card_front", "fields": {"id_number": "79175219700809243X", "name": "谢衍"}, "file": "id1.jpg"},
        {"doc_type": "loan_application", "fields": {"id_number": "791752197108091901", "name": "谢衍",
                                                     "spouse_id_number": "861453198405102155"}, "file": "app.jpg"},
        {"doc_type": "marriage_cert", "fields": {"spouse_id_number": "861453198805101557"}, "file": "marry.jpg"},
        {"doc_type": "bank_statement", "fields": {}, "ocr_text": "USDT 充值 5000", "transactions": [{"date": "2025-01-01"}]},
        {"doc_type": "income_certificate", "fields": {"monthly_income": "8000"}},
        {"doc_type": "credit_report", "fields": {"overdue_total": 8, "overdue_consecutive": 3}},
    ]
    strat = detect_product(docs)["strategy"]
    result = run_all_layers(docs, strat, rule_result={"avg_monthly_income": 5000}, kg_alerts=[])

    print(f"自动动作: {result['auto_action']}  (block={result['block_count']} major={result['major_count']})")
    print(f"分层: {result['layer_summary']}")
    print(f"严重度: {result['severity_summary']}")
    print(f"\n发现 {len(result['findings'])} 条:")
    for f in result['findings']:
        print(f"  [{f['layer']}] {f['rule_id']} {f['severity']:6s} {f['rule_name']}: {f['evidence'][:60]}")
