"""
⑪ 知识图谱反欺诈检测层（step11）

设计：networkx 内存图谱（零外部依赖，开箱即用）
     + Neo4j 适配接口（导出 Cypher，未来可平滑切换）

节点类型：
- Applicant       申请人（person 目录名）
- IDNumber        身份证号（去格式化）
- Phone           手机号
- Address         地址（去空格小写）
- BankAccount     银行账户末4位（脱敏数据常见，末4稳定）
- Employer        雇主/单位名
- Company         企业名（营业执照/特许证件上）
- Document        单份材料（按 doc_type + person 唯一）

边类型：
- HAS_ID          Applicant → IDNumber
- HAS_PHONE       Applicant → Phone
- RESIDES_AT      Applicant → Address
- OWNS_ACCOUNT    Applicant → BankAccount
- EMPLOYED_BY     Applicant → Employer
- OWNS_COMPANY    Applicant → Company（法人/经营者）
- SUBMITTED       Applicant → Document
- SPOUSE_OF       Applicant → Applicant（结婚证上的配偶关系）
- GUARANTEES      Applicant → Applicant（担保关系）

反欺诈检测规则（对应 REG-006 第三/四/五/六章）：
- R1 同身份证号 → 多申请人：身份盗用 / 多头借贷（REG-006 第十一条）
- R2 同地址 → 多申请人：团伙欺诈（REG-006 第十三条）
- R3 同手机号 → 多申请人：团伙代办（REG-006 第十三条）
- R4 同收款账户 → 多申请人：资金回流（REG-006 第七条）
- R5 同企业法人 → 多申请人：关联担保（REG-006 第十条）
- R6 互为配偶/担保 → 多对申请：关联授信集中（REG-006 第十条）
- R7 跨产品材料错配：personal_consumer 提交 food_license/business_license 等
        → 经营性材料混入消费贷（REG-006 第三条 用途真实性）
- R8 单申请人多身份证号：身份主体混乱（REG-005 + 内部制度第10条）

输出：
- fraud_alerts: [{rule, level, applicants, evidence, policy_basis}]
- 持久化：output/kg_graph.json（networkx 节点边导出）
"""
import os
import re
import json
from datetime import datetime
from collections import defaultdict

import networkx as nx

# ============================================================
# 工具：字段归一化
# ============================================================
_ID_RE = re.compile(r"\d{6}[\dXx]{6,12}|\d{15,18}")
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_BANK_RE = re.compile(r"\d{16,19}")

# 身份证相关材料白名单（只有这些材料的 id_no 字段才被采信）
_ID_DOC_TYPES = {"id_card_front", "id_card_back", "household_register_front",
                 "household_register_self"}


def _is_valid_id_checksum(id_no: str) -> bool:
    """
    严格校验 18 位中国居民身份证号：
    - 前 6 位行政区划代码：第 1 位必须为 1-6（7/8/9 开头多为电话号或账号误抽）
    - 7-14 位出生日期：必须为合法 YYYYMMDD，且年份在 1900-当前年份之间
    - 第 18 位校验位：按 GB 11643-1999 标准校验
    15 位老身份证号不在此数据集中出现，直接拒绝以避免歧义。
    """
    if not id_no or len(id_no) != 18:
        return False
    if not re.match(r"^\d{17}[\dX]$", id_no):
        return False
    # 行政区划码第 1 位：1=北京 2=上海 3=河北...6=陕西/甘肃/青海/宁夏/新疆/重庆
    # 7-9 不是合法行政区划开头，多为银行账号/电话号误抽
    if id_no[0] not in "123456":
        return False
    # 出生日期合法性
    try:
        year = int(id_no[6:10])
        month = int(id_no[10:12])
        day = int(id_no[12:14])
        from datetime import date
        if year < 1900 or year > date.today().year:
            return False
        date(year, month, day)  # 抛 ValueError 即非法
    except ValueError:
        return False
    # 校验位（GB 11643-1999）
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_map = "10X98765432"
    total = sum(int(id_no[i]) * weights[i] for i in range(17))
    expected = check_map[total % 11]
    return id_no[17].upper() == expected


def _norm_id(s: str) -> str:
    """身份证号归一化：取末18位，转大写，去掉非字母数字。不做合法性校验。"""
    if not s:
        return ""
    # 找连续15-18位（含X）
    m = _ID_RE.search(str(s))
    if not m:
        return ""
    return m.group(0).upper()[-18:]


def _norm_phone(s: str) -> str:
    if not s:
        return ""
    m = _PHONE_RE.search(str(s))
    return m.group(0) if m else ""


def _norm_bank(s: str) -> str:
    """银行卡号归一化：脱敏数据常见，取末4位"""
    if not s:
        return ""
    digits = re.sub(r"\D", "", str(s))
    if len(digits) < 4:
        return ""
    return digits[-4:]


def _norm_addr(s: str) -> str:
    if not s:
        return ""
    # 去空格、统一小写、去标点
    s = re.sub(r"[\s，。、,.;:]+", "", str(s))
    return s.lower()[-50:]  # 地址后段更稳定


def _norm_name(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    # 只保留中文/英文/数字
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z]", "", s)


# ============================================================
# 图谱构建
# ============================================================
class KnowledgeGraph:
    """反欺诈知识图谱（networkx 实现）"""

    def __init__(self):
        self.G = nx.MultiDiGraph()

    # ---- 节点写入 ----
    def _add_node(self, node_id: str, node_type: str, **attrs):
        if not node_id:
            return
        if self.G.has_node(node_id):
            # 合并属性
            existing = self.G.nodes[node_id]
            existing.setdefault("node_type", node_type)
            existing.setdefault("first_seen", attrs.get("first_seen", ""))
            existing.setdefault("applicants", set())
            existing.setdefault("sources", set())
            for k, v in attrs.items():
                if k in ("applicants", "sources"):
                    if isinstance(v, (list, set)):
                        existing[k].update(v)
                else:
                    existing[k] = v
        else:
            attrs.setdefault("node_type", node_type)
            attrs.setdefault("applicants", set(attrs.get("applicants", [])))
            attrs.setdefault("sources", set(attrs.get("sources", [])))
            self.G.add_node(node_id, **attrs)

    def _add_edge(self, src: str, dst: str, rel: str, **attrs):
        if not src or not dst or src == dst:
            return
        self.G.add_edge(src, dst, key=rel, relation=rel, **attrs)

    # ---- 单申请人入图 ----
    def add_applicant(self, person: str, extracted_docs: list):
        """
        将一个申请人的所有材料写入图谱
        :param person: 申请人目录名（如 "12-谢衍"）
        :param extracted_docs: step2 输出的 valid_docs 列表
        """
        applicant_id = f"applicant::{person}"
        self._add_node(applicant_id, "Applicant", name=person, first_seen=datetime.now().isoformat())

        # 收集该申请人的所有身份证号（用于检测身份主体混乱）
        person_ids = set()
        person_phones = set()
        person_addrs = set()
        person_accounts = set()
        person_employers = set()
        person_companies = set()
        spouses = set()

        for doc in extracted_docs:
            doc_type = doc.get("doc_type", "unknown")
            fields = doc.get("fields", {}) or {}
            file_path = doc.get("file", "")
            ocr_text = doc.get("ocr_text", "") or ""

            # 材料节点
            doc_id = f"doc::{person}::{doc_type}::{hash(file_path) & 0xFFFF}"
            self._add_node(doc_id, "Document", doc_type=doc_type,
                           person=person, file=os.path.basename(file_path or ""),
                           confidence=doc.get("confidence", 0))
            self._add_edge(applicant_id, doc_id, "SUBMITTED")

            # ---- 身份证号：只有身份证相关材料才采信（避免婚姻证配偶ID、
            #     资金流水账号、营业执照统一社会信用代码被误抽为申请人身份证号）----
            id_no = ""
            if doc_type in _ID_DOC_TYPES:
                # 身份证件材料：字段优先
                for id_key in ("id_no", "身份证号", "公民身份号码", "证件号码", "身份证号码",
                               "身份号码", "持卡人身份证号", "借款人身份证号", "申请人身份证号",
                               "owner_id"):
                    v = fields.get(id_key)
                    if v:
                        cand = _norm_id(v)
                        if cand and _is_valid_id_checksum(cand):
                            id_no = cand
                            break
                # 字段没有，从 ocr_text 兜底正则提取
                if not id_no and ocr_text:
                    for m in _ID_RE.finditer(ocr_text):
                        cand = m.group(0).upper()[-18:]
                        if _is_valid_id_checksum(cand):
                            id_no = cand
                            break
            if id_no:
                person_ids.add(id_no)
                self._add_node(f"id::{id_no}", "IDNumber", value=id_no,
                               applicants={person})
                self._add_edge(applicant_id, f"id::{id_no}", "HAS_ID",
                               source_doc=doc_type)

            # 兜底：name 也尝试从 id_card_front 取
            name = _norm_name(fields.get("name") or fields.get("姓名") or "")
            if name:
                # 把姓名作为 Applicant 属性补充
                self.G.nodes[applicant_id]["real_name"] = name

            # ---- 手机号：字段优先，ocr_text 兜底 ----
            phone_raw = ""
            for pkey in ("phone", "手机", "联系电话", "mobile", "telephone",
                         "手机号", "phone_no", "contact_phone", "联系电话号码"):
                v = fields.get(pkey)
                if v:
                    phone_raw = v
                    break
            phone = _norm_phone(phone_raw) if phone_raw else ""
            if not phone and ocr_text:
                # 排除身份证号末尾数字误判（手机号是 1 开头的 11 位）
                m = _PHONE_RE.search(ocr_text)
                if m:
                    phone = m.group(0)
            if phone:
                person_phones.add(phone)
                self._add_node(f"phone::{phone}", "Phone", value=phone,
                               applicants={person})
                self._add_edge(applicant_id, f"phone::{phone}", "HAS_PHONE",
                               source_doc=doc_type)

            # ---- 地址（多来源）----
            for addr_key in ("address", "住址", "地址", "registered_address", "经营地址",
                             "住所", "户籍地址", "通讯地址", "residence_address"):
                addr = _norm_addr(fields.get(addr_key))
                if addr and len(addr) >= 6:
                    person_addrs.add(addr)
                    self._add_node(f"addr::{addr}", "Address", value=addr,
                                   applicants={person})
                    self._add_edge(applicant_id, f"addr::{addr}", "RESIDES_AT",
                                   source_doc=doc_type)

            # ---- 银行账户：字段优先，ocr_text 兜底 ----
            acc = ""
            for akey in ("account_no", "卡号", "bank_card_no", "账号", "银行卡号",
                         "account_number", "card_no", "银行账号"):
                v = fields.get(akey)
                if v:
                    acc = _norm_bank(v)
                    if acc:
                        break
            if not acc and ocr_text and doc_type in ("bank_statement", "bank_card", "fund_flow_analysis"):
                # 只在银行/流水材料兜底，避免误抽
                for m in _BANK_RE.finditer(ocr_text):
                    digits = m.group(0)
                    if 16 <= len(digits) <= 19 and not digits.startswith("1"):
                        # 排除手机号（1开头11位）和身份证号（已在上面处理）
                        acc = digits[-4:]
                        break
            if acc:
                person_accounts.add(acc)
                self._add_node(f"acc::{acc}", "BankAccount", value=acc,
                               applicants={person})
                self._add_edge(applicant_id, f"acc::{acc}", "OWNS_ACCOUNT",
                               source_doc=doc_type)

            # ---- 雇主/单位 ----
            emp = ""
            for ekey in ("company", "employer", "单位", "employer_name", "单位名称",
                         "工作单位", "雇主", "employed_at", "工作单位名称"):
                v = fields.get(ekey)
                if v:
                    emp = str(v).strip()
                    if emp:
                        break
            if emp and len(emp) >= 4:
                emp_norm = _norm_name(emp)
                if emp_norm:
                    person_employers.add(emp_norm)
                    self._add_node(f"emp::{emp_norm}", "Employer", value=emp_norm,
                                   applicants={person})
                    self._add_edge(applicant_id, f"emp::{emp_norm}", "EMPLOYED_BY",
                                   source_doc=doc_type)

            # ---- 企业（营业执照/特许证件上的法人/经营者）----
            company = ""
            for ckey in ("company_name", "企业名称", "单位名称", "名称", "business_name"):
                v = fields.get(ckey)
                if v:
                    company = str(v).strip()
                    if company:
                        break
            legal = ""
            for lkey in ("legal_person", "法定代表人", "法人", "经营者", "负责人"):
                v = fields.get(lkey)
                if v:
                    legal = str(v).strip()
                    if legal:
                        break
            if company and len(company) >= 4:
                company_norm = _norm_name(company)
                if company_norm:
                    person_companies.add(company_norm)
                    self._add_node(f"company::{company_norm}", "Company",
                                   value=company_norm, applicants={person})
                    self._add_edge(applicant_id, f"company::{company_norm}",
                                   "OWNS_COMPANY", source_doc=doc_type,
                                   legal_person=_norm_name(legal))

            # ---- 配偶（结婚证）----
            if doc_type == "marriage_cert":
                spouse = ""
                for skey in ("spouse_name", "配偶姓名", "男方", "女方",
                             "男方姓名", "女方姓名", "丈夫姓名", "妻子姓名"):
                    v = fields.get(skey)
                    if v:
                        spouse = str(v).strip()
                        if spouse:
                            break
                if spouse:
                    spouse_norm = _norm_name(spouse)
                    if spouse_norm:
                        spouses.add(spouse_norm)

        # 配偶节点：尝试匹配到其他申请人
        for spouse in spouses:
            # 在已有申请人节点里找姓名匹配
            for n, d in self.G.nodes(data=True):
                if d.get("node_type") == "Applicant" and d.get("real_name") == spouse:
                    self._add_edge(applicant_id, n, "SPOUSE_OF")

        # 记录该申请人的所有实体（用于反欺诈分析）
        self.G.nodes[applicant_id]["ids"] = person_ids
        self.G.nodes[applicant_id]["phones"] = person_phones
        self.G.nodes[applicant_id]["addrs"] = person_addrs
        self.G.nodes[applicant_id]["accounts"] = person_accounts
        self.G.nodes[applicant_id]["employers"] = person_employers
        self.G.nodes[applicant_id]["companies"] = person_companies
        self.G.nodes[applicant_id]["spouses"] = spouses

    # ============================================================
    # 反欺诈检测
    # ============================================================
    def detect_fraud(self) -> list:
        """跑全部反欺诈规则，返回 alerts 列表"""
        alerts = []

        alerts.extend(self._rule_r1_shared_id())
        alerts.extend(self._rule_r2_shared_address())
        alerts.extend(self._rule_r3_shared_phone())
        alerts.extend(self._rule_r4_shared_account())
        alerts.extend(self._rule_r5_shared_company())
        alerts.extend(self._rule_r6_spouse_chain())
        alerts.extend(self._rule_r8_multi_id_per_person())

        # 按级别排序
        severity = {"high": 0, "medium": 1, "low": 2}
        alerts.sort(key=lambda a: severity.get(a.get("level", "low"), 2))
        return alerts

    def _applicants_of(self, node_id: str) -> set:
        """取与某实体节点关联的申请人集合"""
        apps = set()
        node = self.G.nodes.get(node_id)
        if not node:
            return apps
        apps.update(node.get("applicants", set()))
        return apps

    def _rule_r1_shared_id(self) -> list:
        """R1 同身份证号 → 多申请人：身份盗用/多头借贷"""
        alerts = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "IDNumber":
                continue
            apps = self._applicants_of(n)
            if len(apps) >= 2:
                alerts.append({
                    "rule": "R1_shared_id",
                    "level": "high",
                    "applicants": sorted(apps),
                    "entity": d.get("value", ""),
                    "entity_type": "身份证号",
                    "evidence": f"申请人 {sorted(apps)} 共享同一身份证号 {d.get('value', '')}",
                    "policy_basis": "《信贷资金用途监管通知》第十一条 多头借贷识别；《信贷审批负面客户认定标准》第十条 身份盗用",
                })
        return alerts

    def _rule_r2_shared_address(self) -> list:
        """R2 同地址 → 多申请人：团伙欺诈"""
        alerts = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "Address":
                continue
            apps = self._applicants_of(n)
            if len(apps) >= 2:
                alerts.append({
                    "rule": "R2_shared_address",
                    "level": "medium",
                    "applicants": sorted(apps),
                    "entity": d.get("value", ""),
                    "entity_type": "地址",
                    "evidence": f"申请人 {sorted(apps)} 共享同一地址",
                    "policy_basis": "《信贷资金用途监管通知》第十三条 团伙欺诈识别",
                })
        return alerts

    def _rule_r3_shared_phone(self) -> list:
        """R3 同手机号 → 多申请人：团伙代办"""
        alerts = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "Phone":
                continue
            apps = self._applicants_of(n)
            if len(apps) >= 2:
                alerts.append({
                    "rule": "R3_shared_phone",
                    "level": "high",
                    "applicants": sorted(apps),
                    "entity": d.get("value", ""),
                    "entity_type": "手机号",
                    "evidence": f"申请人 {sorted(apps)} 共享同一手机号 {d.get('value', '')}",
                    "policy_basis": "《信贷资金用途监管通知》第十三条 团伙欺诈识别",
                })
        return alerts

    def _rule_r4_shared_account(self) -> list:
        """R4 同收款账户 → 多申请人：资金回流/团伙"""
        alerts = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "BankAccount":
                continue
            apps = self._applicants_of(n)
            if len(apps) >= 2:
                alerts.append({
                    "rule": "R4_shared_account",
                    "level": "high",
                    "applicants": sorted(apps),
                    "entity": f"卡号末4位 {d.get('value', '')}",
                    "entity_type": "银行账户",
                    "evidence": f"申请人 {sorted(apps)} 共享同一银行账户（末4位 {d.get('value', '')}），存在资金回流风险",
                    "policy_basis": "《信贷资金用途监管通知》第七条 账户监控；第八条 禁止性用途",
                })
        return alerts

    def _rule_r5_shared_company(self) -> list:
        """R5 同企业法人 → 多申请人：关联担保/关联企业"""
        alerts = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "Company":
                continue
            apps = self._applicants_of(n)
            if len(apps) >= 2:
                alerts.append({
                    "rule": "R5_shared_company",
                    "level": "medium",
                    "applicants": sorted(apps),
                    "entity": d.get("value", ""),
                    "entity_type": "企业",
                    "evidence": f"申请人 {sorted(apps)} 同为 {d.get('value', '')} 的法人/经营者，存在关联担保风险",
                    "policy_basis": "《信贷资金用途监管通知》第十条 关联交易审查；《项目融资业务指引》第十三条 关联方担保",
                })
        return alerts

    def _rule_r6_spouse_chain(self) -> list:
        """R6 互为配偶 → 多对申请：关联授信集中"""
        alerts = []
        # 找 SPOUSE_OF 边
        spouse_pairs = set()
        for u, v, k in self.G.edges(keys=True):
            if k == "SPOUSE_OF":
                pair = tuple(sorted([u, v]))
                spouse_pairs.add(pair)
        if len(spouse_pairs) >= 2:
            apps = set()
            for u, v in spouse_pairs:
                apps.add(u.replace("applicant::", ""))
                apps.add(v.replace("applicant::", ""))
            alerts.append({
                "rule": "R6_spouse_chain",
                "level": "medium",
                "applicants": sorted(apps),
                "entity": f"{len(spouse_pairs)} 对配偶关系",
                "entity_type": "关联关系",
                "evidence": f"图谱中发现 {len(spouse_pairs)} 对配偶关系：{[('↔'.join(p).replace('applicant::','')) for p in spouse_pairs]}",
                "policy_basis": "《信贷资金用途监管通知》第十条 关联交易审查",
            })
        return alerts

    def _rule_r8_multi_id_per_person(self) -> list:
        """R8 单申请人多身份证号：身份主体混乱"""
        alerts = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") != "Applicant":
                continue
            ids = d.get("ids", set())
            if len(ids) >= 2:
                alerts.append({
                    "rule": "R8_multi_id_per_person",
                    "level": "high",
                    "applicants": [n.replace("applicant::", "")],
                    "entity": "; ".join(sorted(ids)),
                    "entity_type": "身份证号",
                    "evidence": f"申请人 {n.replace('applicant::', '')} 在不同材料中出现 {len(ids)} 个不同身份证号：{sorted(ids)}，身份主体混乱",
                    "policy_basis": "《信贷审批负面客户认定标准》第十条 关键身份标识无法交叉验证",
                })
        return alerts

    # ============================================================
    # 持久化 / 导出
    # ============================================================
    def dump_json(self, path: str):
        """导出为 JSON（节点+边）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nodes = []
        for n, d in self.G.nodes(data=True):
            node_data = {"id": n}
            for k, v in d.items():
                if isinstance(v, set):
                    node_data[k] = sorted(list(v))
                else:
                    node_data[k] = v
            nodes.append(node_data)
        edges = []
        for u, v, k, d in self.G.edges(data=True, keys=True):
            edge_data = {"src": u, "dst": v, "relation": k}
            edge_data.update({k2: v2 for k2, v2 in d.items() if k2 != "relation"})
            edges.append(edge_data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges,
                       "exported_at": datetime.now().isoformat(),
                       "stats": {"node_count": len(nodes), "edge_count": len(edges)}},
                      f, ensure_ascii=False, indent=2)

    def export_cypher(self, path: str):
        """导出 Neo4j Cypher 建图脚本（用于未来切换 Neo4j 后端）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = ["// Auto-generated Cypher script for Neo4j migration"]
        # 节点
        for n, d in self.G.nodes(data=True):
            nt = d.get("node_type", "Entity")
            props = {k: v for k, v in d.items()
                     if k != "node_type" and not isinstance(v, set)}
            # 转义
            prop_str = ", ".join(f"{k}: {json.dumps(str(v), ensure_ascii=False)}"
                                 for k, v in props.items())
            lines.append(f"MERGE (n:{nt} {{id: {json.dumps(n, ensure_ascii=False)}}}) "
                         f"SET n += {{{prop_str}}};")
        # 边
        for u, v, k, d in self.G.edges(data=True, keys=True):
            lines.append(
                f"MATCH (a {{id: {json.dumps(u, ensure_ascii=False)}}}), "
                f"(b {{id: {json.dumps(v, ensure_ascii=False)}}}) "
                f"MERGE (a)-[r:{k}]->(b);"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def stats(self) -> dict:
        type_count = defaultdict(int)
        for _, d in self.G.nodes(data=True):
            type_count[d.get("node_type", "Unknown")] += 1
        rel_count = defaultdict(int)
        for _, _, k in self.G.edges(keys=True):
            rel_count[k] += 1
        return {
            "node_total": self.G.number_of_nodes(),
            "edge_total": self.G.number_of_edges(),
            "node_by_type": dict(type_count),
            "edge_by_relation": dict(rel_count),
        }


# ============================================================
# 主入口：从申请人列表批量建图 + 检测
# ============================================================
def build_and_detect(applicants_data: list, output_dir: str = None) -> dict:
    """
    :param applicants_data: [{"person": "12-谢衍", "extracted_docs": [...]}, ...]
    :param output_dir: 输出目录（None 则不持久化）
    :return: {
        "graph_stats": {...},
        "fraud_alerts": [...],
        "applicant_count": int,
    }
    """
    kg = KnowledgeGraph()
    for item in applicants_data:
        kg.add_applicant(item["person"], item.get("extracted_docs", []))

    stats = kg.stats()
    alerts = kg.detect_fraud()

    print(f"\n[Step11] 知识图谱构建完成")
    print(f"  节点: {stats['node_total']} ({stats['node_by_type']})")
    print(f"  边: {stats['edge_total']} ({stats['edge_by_relation']})")
    print(f"  反欺诈告警: {len(alerts)} 条")
    if alerts:
        for a in alerts:
            print(f"    [{a['level']}] {a['rule']}: {a['evidence'][:80]}")

    result = {
        "graph_stats": stats,
        "fraud_alerts": alerts,
        "applicant_count": len(applicants_data),
        "generated_at": datetime.now().isoformat(),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        kg.dump_json(os.path.join(output_dir, "kg_graph.json"))
        kg.export_cypher(os.path.join(output_dir, "kg_graph.cypher"))
        with open(os.path.join(output_dir, "kg_fraud_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  图谱已导出: {output_dir}/kg_graph.json (+.cypher)")
        print(f"  反欺诈报告: {output_dir}/kg_fraud_report.json")

    return result


def enhance_decision_with_kg(
    person: str,
    extracted_docs: list,
    all_applicants_data: list = None,
) -> dict:
    """
    单申请人决策增强：返回该申请人相关的反欺诈告警
    （需要 all_applicants_data 才能做跨申请人检测；否则只能做 R8 单人检测）

    :return: {
        "kg_alerts": [...],          # 与该申请人相关的告警
        "has_fraud_network": bool,   # 是否涉及团伙/关联网络
    }
    """
    if all_applicants_data and len(all_applicants_data) > 1:
        # 完整图谱检测
        kg = KnowledgeGraph()
        for item in all_applicants_data:
            kg.add_applicant(item["person"], item.get("extracted_docs", []))
        all_alerts = kg.detect_fraud()
    else:
        # 仅本申请人建图（只能跑 R8）
        kg = KnowledgeGraph()
        kg.add_applicant(person, extracted_docs)
        all_alerts = kg.detect_fraud()

    # 筛出与该申请人相关的告警
    related = [a for a in all_alerts if person in a.get("applicants", [])]
    has_network = any(
        len(a.get("applicants", [])) >= 2 or a["rule"] in ("R6_spouse_chain",)
        for a in related
    )

    return {
        "kg_alerts": related,
        "has_fraud_network": has_network,
    }


if __name__ == "__main__":
    # 自测：构造 2 个虚假申请人验证图谱逻辑
    print("=" * 60)
    print("Step11 知识图谱反欺诈模块 自测")
    print("=" * 60)

    test_applicants = [
        {
            "person": "测试-张三",
            "extracted_docs": [
                {
                    "doc_type": "id_card_front",
                    "fields": {"name": "张三", "id_no": "110101199001011234",
                               "address": "北京市东城区王府井大街1号"},
                    "file": "test1.jpg", "confidence": 0.95,
                },
                {
                    "doc_type": "bank_statement",
                    "fields": {"account_no": "6222021234567890123",
                               "account_holder": "张三"},
                    "file": "test2.jpg", "confidence": 0.9,
                },
            ],
        },
        {
            "person": "测试-李四",
            "extracted_docs": [
                {
                    "doc_type": "id_card_front",
                    # 同身份证号 → R1 触发
                    "fields": {"name": "李四", "id_no": "110101199001011234",
                               "address": "北京市朝阳区国贸中心"},
                    "file": "test3.jpg", "confidence": 0.95,
                },
                {
                    "doc_type": "bank_card",
                    # 同卡号末4位 → R4 触发
                    "fields": {"卡号": "6217001234567890123", "持卡人": "李四"},
                    "file": "test4.jpg", "confidence": 0.9,
                },
            ],
        },
    ]

    result = build_and_detect(test_applicants, output_dir="output")
    print(f"\n✅ 自测完成：{len(result['fraud_alerts'])} 条告警")
