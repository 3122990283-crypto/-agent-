"""
RAG 知识检索层（step10）

三库分 collection：
- regulation：监管法规库（上位法，最高权重）
- policy_regulation：内部制度库（核心依据）
- case_reference：疑难案例汇编（参考层，最低权重）
- structured_rules：结构化准入规则（精确匹配，不走向量）

检索策略：
- 向量检索（BGE-M3）用于条款语义召回
- 结构化字段精确匹配用于准入阈值（如 DTI/逾期次数）
- prompt 层级约束：法规 > 制度 > 案例
"""
import os
import json
import re
import math
import glob
from typing import Optional

# chromadb / FlagEmbedding 为可选依赖，不可用时降级到纯 BM25 文件检索
try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except ImportError:
    chromadb = None
    Settings = None
    _HAS_CHROMA = False

try:
    from FlagEmbedding import FlagModel
    _HAS_FLAGEMB = True
except ImportError:
    FlagModel = None
    _HAS_FLAGEMB = False

# 知识库根目录
KB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
POLICY_DIR = os.path.join(KB_ROOT, "policies")
REGULATION_DIR = os.path.join(KB_ROOT, "regulations")
DB_DIR = os.path.join(KB_ROOT, "chroma_db")

# BGE-M3 模型（首次使用自动下载）
_EMBEDDER: Optional["FlagModel"] = None


# ============================================================
# 纯 Python BM25 实现（无需外部库，字符级 bigram 分词）
# ============================================================
def _tokenize_zh(text: str) -> list:
    """中文分词：字符级 bigram + 单字 + 数字/英文词组"""
    if not text:
        return []
    # 提取中文连续段、数字、英文词
    tokens = []
    # 数字串
    for m in re.finditer(r"\d+", text):
        tokens.append(m.group(0))
    # 英文词
    for m in re.finditer(r"[A-Za-z]+", text):
        tokens.append(m.group(0).lower())
    # 中文字符 bigram
    chinese = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    for i in range(len(chinese) - 1):
        tokens.append(chinese[i:i + 2])
    # 单字（提升短查询召回）
    for c in chinese:
        tokens.append(c)
    return tokens


class BM25Index:
    """纯 Python BM25Okapi 实现（无外部依赖）"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []          # 原文
        self.doc_tokens = []    # 分词后
        self.doc_len = []
        self.avgdl = 0
        self.df = {}            # 词 -> 出现文档数
        self.idf = {}
        self.tf = []            # 每文档的词频
        self.N = 0

    def add_docs(self, docs: list):
        """批量加入文档（docs: [{content, ...}]）"""
        for d in docs:
            content = d.get("content", "")
            tokens = _tokenize_zh(content)
            self.docs.append(d)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            # 词频
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            # df
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)
        if self.doc_len:
            self.avgdl = sum(self.doc_len) / self.N
        # 计算 idf
        for t, df in self.df.items():
            # BM25Okapi idf 公式（加 0.5 平滑）
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> list:
        """检索 top_k 文档，返回 [{content, source, article, score}]"""
        if self.N == 0:
            return []
        q_tokens = _tokenize_zh(query)
        scores = []
        for i in range(self.N):
            s = 0.0
            tf = self.tf[i]
            dl = self.doc_len[i]
            for t in q_tokens:
                if t not in self.idf:
                    continue
                if t not in tf:
                    continue
                f = tf[t]
                idf = self.idf[t]
                # BM25 公式
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        results = []
        for i, s in scores[:top_k]:
            if s <= 0:
                continue
            d = self.docs[i]
            results.append({
                "content": d.get("content", ""),
                "source": d.get("source", "") or d.get("metadata", {}).get("source", ""),
                "chapter": d.get("chapter", "") or d.get("metadata", {}).get("chapter", ""),
                "article": d.get("article", "") or d.get("metadata", {}).get("article", ""),
                "layer": d.get("layer", "") or d.get("metadata", {}).get("layer", ""),
                "score": round(s, 4),
            })
        return results


# 三层 BM25 索引（按 collection 隔离）
_BM25_INDEXES: dict = {}  # {collection_name: BM25Index}


def _load_bm25_from_chroma(client) -> None:
    """
    从 Chroma 持久化的三个 collection 读取全量文档，重建 BM25 索引。
    进程重启后 _BM25_INDEXES 会丢失，调用本函数可免重新 embedding 恢复 Hybrid 检索能力。
    幂等：已加载则跳过。
    """
    global _BM25_INDEXES
    if _BM25_INDEXES:
        return
    print("[RAG] 从 Chroma 恢复 BM25 索引...")
    for col_name in ("regulation", "policy_regulation", "case_reference"):
        try:
            col = client.get_collection(col_name)
            if col.count() == 0:
                continue
            data = col.get(include=["documents", "metadatas"])
            docs = []
            for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
                docs.append({
                    "content": doc,
                    "source": meta.get("source", ""),
                    "chapter": meta.get("chapter", ""),
                    "article": meta.get("article", ""),
                    "layer": meta.get("layer", ""),
                })
            if docs:
                idx = BM25Index()
                idx.add_docs(docs)
                _BM25_INDEXES[col_name] = idx
                print(f"  BM25[{col_name}]: {idx.N} 篇文档索引恢复")
        except Exception as e:
            print(f"  BM25[{col_name}] 恢复失败: {e}")


def _load_bm25_from_files() -> None:
    """
    无 chromadb 时从 markdown 文件直接构建 BM25 索引（纯文件检索降级方案）。
    幂等：已加载则跳过。
    """
    global _BM25_INDEXES
    if _BM25_INDEXES:
        return
    print("[RAG] chromadb 不可用，从 markdown 文件构建 BM25 索引...")

    # 1. 法规库
    reg_docs = []
    for md_file in sorted(glob.glob(os.path.join(REGULATION_DIR, "*.md"))):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        for chunk in _split_policy_markdown(content, md_file, layer="regulation"):
            reg_docs.append({"content": chunk["content"], **chunk["metadata"]})
    if reg_docs:
        idx = BM25Index()
        idx.add_docs(reg_docs)
        _BM25_INDEXES["regulation"] = idx
        print(f"  BM25[regulation]: {idx.N} 篇文档索引完成（来自文件）")

    # 2. 制度库 + 3. 案例库
    policy_docs = []
    case_docs = []
    for md_file in sorted(glob.glob(os.path.join(POLICY_DIR, "*.md"))):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        is_case = "案例" in os.path.basename(md_file)
        layer = "case" if is_case else "policy"
        for chunk in _split_policy_markdown(content, md_file, layer=layer):
            target = case_docs if is_case else policy_docs
            target.append({"content": chunk["content"], **chunk["metadata"]})
    if policy_docs:
        idx = BM25Index()
        idx.add_docs(policy_docs)
        _BM25_INDEXES["policy_regulation"] = idx
        print(f"  BM25[policy_regulation]: {idx.N} 篇文档索引完成（来自文件）")
    if case_docs:
        idx = BM25Index()
        idx.add_docs(case_docs)
        _BM25_INDEXES["case_reference"] = idx
        print(f"  BM25[case_reference]: {idx.N} 篇文档索引完成（来自文件）")


def _resolve_bge_m3_path() -> str:
    """优先用本地 HF 缓存快照路径，避免 hub 网络检查超时"""
    import glob as _glob
    # 1. 环境变量显式指定
    env_path = os.environ.get("BGE_M3_PATH")
    if env_path and os.path.exists(os.path.join(env_path, "config.json")):
        return env_path
    # 2. HF 缓存快照（Windows: %USERPROFILE%\.cache\huggingface\hub）
    cache_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    snap_glob = os.path.join(cache_root, "models--BAAI--bge-m3", "snapshots", "*")
    for s in sorted(_glob.glob(snap_glob)):
        if os.path.exists(os.path.join(s, "config.json")):
            return s
    # 3. fallback 到 repo id（需联网）
    return "BAAI/bge-m3"


def _get_embedder() -> FlagModel:
    global _EMBEDDER
    if _EMBEDDER is None:
        # 离线模式：BGE-M3 已缓存到 ~/.cache/huggingface，跳过 hub 网络检查避免超时
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model_path = _resolve_bge_m3_path()
        print(f"[RAG] 加载 BGE-M3 embedding 模型: {model_path}")
        _EMBEDDER = FlagModel(
            model_path,
            use_fp16=True,
        )
        print("[RAG] BGE-M3 加载完成")
    return _EMBEDDER


# ============================================================
# 文档分块（按条款分，保留章节上下文）
# ============================================================
def _split_policy_markdown(content: str, source_file: str, layer: str = "policy") -> list:
    """
    将制度 Markdown 按条款分块。
    每个"第X条"是一个 chunk，保留所属章节作为上下文。
    :param layer: regulation(法规)/policy(制度)/case(案例)
    """
    chunks = []
    # 匹配章节标题和条款
    chapter_pattern = re.compile(r"^##\s+(第.章.+)$", re.MULTILINE)
    article_pattern = re.compile(r"^###\s+(第.+条.+)$", re.MULTILINE)

    # 找所有章节
    chapters = list(chapter_pattern.finditer(content))
    chapter_ranges = []
    for i, m in enumerate(chapters):
        start = m.start()
        end = chapters[i + 1].start() if i + 1 < len(chapters) else len(content)
        chapter_ranges.append((m.group(1), start, end))

    # 找所有条款
    articles = list(article_pattern.finditer(content))
    for i, art in enumerate(articles):
        art_start = art.start()
        art_end = articles[i + 1].start() if i + 1 < len(articles) else len(content)
        art_content = content[art_start:art_end].strip()

        # 找所属章节
        chapter_title = ""
        for ch_title, ch_start, ch_end in chapter_ranges:
            if ch_start <= art_start < ch_end:
                chapter_title = ch_title
                break

        # 提取条款编号（如"第一条"）
        article_title = art.group(1)

        chunks.append({
            "id": f"{os.path.basename(source_file)}::{article_title}",
            "content": f"【{chapter_title}】\n{art_content}" if chapter_title else art_content,
            "metadata": {
                "source": os.path.basename(source_file),
                "chapter": chapter_title,
                "article": article_title,
                "layer": layer,
            },
        })

    # 如果没有按条款分（如案例汇编），按 ## 标题分块
    if not chunks:
        section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        sections = list(section_pattern.finditer(content))
        if sections:
            for i, sec in enumerate(sections):
                sec_start = sec.start()
                sec_end = sections[i + 1].start() if i + 1 < len(sections) else len(content)
                sec_content = content[sec_start:sec_end].strip()
                chunks.append({
                    "id": f"{os.path.basename(source_file)}::{sec.group(1)}",
                    "content": sec_content,
                    "metadata": {
                        "source": os.path.basename(source_file),
                        "chapter": sec.group(1),
                        "article": "",
                        "layer": layer,
                    },
                })

    # 兜底：整篇作为一个 chunk
    if not chunks:
        chunks.append({
            "id": os.path.basename(source_file),
            "content": content,
            "metadata": {
                "source": os.path.basename(source_file),
                "chapter": "",
                "article": "",
                "layer": layer,
            },
        })

    return chunks


# ============================================================
# 知识库构建
# ============================================================
def build_knowledge_base():
    """
    构建三层知识库：
    - regulation collection：监管法规库（最高权重）
    - policy_regulation collection：内部制度库（核心依据）
    - case_reference collection：案例库（参考层）
    chromadb / FlagEmbedding 不可用时仅构建 BM25 文件索引。
    """
    print("[RAG] 开始构建知识库（三层）...")

    # 降级路径：chromadb 或 FlagEmbedding 不可用 → 仅构建 BM25 索引
    if not _HAS_CHROMA or not _HAS_FLAGEMB:
        print("[RAG] chromadb/FlagEmbedding 不可用，仅构建 BM25 文件索引")
        _load_bm25_from_files()
        return None

    embedder = _get_embedder()

    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # 删除旧 collection 重建
    for col_name in ["regulation", "policy_regulation", "case_reference"]:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    reg_col = client.get_or_create_collection(
        name="regulation",
        metadata={"hnsw:space": "cosine"},
    )
    policy_col = client.get_or_create_collection(
        name="policy_regulation",
        metadata={"hnsw:space": "cosine"},
    )
    case_col = client.get_or_create_collection(
        name="case_reference",
        metadata={"hnsw:space": "cosine"},
    )

    # 批量 embedding（FlagEmbedding 新版 API）
    def _embed_batch(chunks):
        texts = [c["content"] for c in chunks]
        embeddings = embedder.encode(texts, batch_size=8)
        # 手动 L2 归一化（新版 encode 不接受 normalize_embedding 参数）
        import numpy as _np
        if isinstance(embeddings, _np.ndarray):
            norms = _np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            return embeddings.tolist()
        return [e.tolist() for e in embeddings]

    # ---- 1. 监管法规库（最高权重）----
    reg_files = glob.glob(os.path.join(REGULATION_DIR, "*.md"))
    total_reg = 0
    reg_bm25_docs = []
    for md_file in reg_files:
        print(f"  [法规] 处理: {os.path.basename(md_file)}")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = _split_policy_markdown(content, md_file, layer="regulation")
        if not chunks:
            continue
        emb_list = _embed_batch(chunks)
        reg_col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            embeddings=emb_list,
            metadatas=[c["metadata"] for c in chunks],
        )
        total_reg += len(chunks)
        print(f"    → {len(chunks)} 条款入库（法规库）")
        # 收集 BM25 文档
        for c in chunks:
            reg_bm25_docs.append({"content": c["content"], **c["metadata"]})

    # ---- 2. 内部制度库 + 案例库 ----
    md_files = glob.glob(os.path.join(POLICY_DIR, "*.md"))
    total_policy = 0
    total_case = 0
    policy_bm25_docs = []
    case_bm25_docs = []
    for md_file in md_files:
        print(f"  [制度/案例] 处理: {os.path.basename(md_file)}")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        is_case = "案例" in os.path.basename(md_file)
        layer = "case" if is_case else "policy"
        chunks = _split_policy_markdown(content, md_file, layer=layer)
        if not chunks:
            continue
        emb_list = _embed_batch(chunks)
        col = case_col if is_case else policy_col
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            embeddings=emb_list,
            metadatas=[c["metadata"] for c in chunks],
        )
        if is_case:
            total_case += len(chunks)
            print(f"    → {len(chunks)} 条款入库（案例库）")
            for c in chunks:
                case_bm25_docs.append({"content": c["content"], **c["metadata"]})
        else:
            total_policy += len(chunks)
            print(f"    → {len(chunks)} 条款入库（制度库）")
            for c in chunks:
                policy_bm25_docs.append({"content": c["content"], **c["metadata"]})

    print(f"\n[RAG] 知识库构建完成")
    print(f"  法规库: {reg_col.count()} 条")
    print(f"  制度库: {policy_col.count()} 条")
    print(f"  案例库: {case_col.count()} 条")

    # ---- 3. 构建 BM25 索引（Hybrid 检索备用）----
    global _BM25_INDEXES
    _BM25_INDEXES = {}
    for name, docs in [
        ("regulation", reg_bm25_docs),
        ("policy_regulation", policy_bm25_docs),
        ("case_reference", case_bm25_docs),
    ]:
        if docs:
            idx = BM25Index()
            idx.add_docs(docs)
            _BM25_INDEXES[name] = idx
            print(f"  BM25[{name}]: {idx.N} 篇文档索引完成")

    # Chroma 1.5 已知问题：写入后需重新打开 client 才能查询
    del client
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client


# ============================================================
# 检索
# ============================================================
def retrieve(
    query: str,
    client=None,
    top_k_regulation: int = 3,
    top_k_policy: int = 5,
    top_k_case: int = 3,
) -> dict:
    """
    Hybrid 检索相关知识（三层）— Dense(向量 BGE-M3) + Sparse(BM25) 用 RRF 融合
    chromadb / FlagEmbedding 不可用时自动降级为纯 BM25 检索。
    :param query: 检索文本（如风险点描述、规则命中信息）
    :return: {
        "regulation_chunks": [{content, source, article, score, rrf_score, score_dense, score_sparse}],   # 法规（最高权重）
        "policy_chunks":     [...],       # 制度
        "case_chunks":       [...],       # 案例
    }
    排序依据 rrf_score（融合后排名）；score 字段保留向量相似度便于展示；
    纯 BM25 命中（向量未召回）的 chunk 用 BM25 归一化分填充 score。
    """
    # ---- 降级路径：chromadb 或 FlagEmbedding 不可用 → 纯 BM25 ----
    if not _HAS_CHROMA or not _HAS_FLAGEMB:
        if not _BM25_INDEXES:
            _load_bm25_from_files()
        return _retrieve_bm25_only(query, top_k_regulation, top_k_policy, top_k_case)

    # ---- 正常路径：Hybrid 检索 ----
    if client is None:
        client = chromadb.PersistentClient(
            path=DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

    # Lazy 加载 BM25 索引（进程重启后 _BM25_INDEXES 为空，从 Chroma 恢复）
    if not _BM25_INDEXES:
        try:
            _load_bm25_from_chroma(client)
        except Exception as e:
            print(f"[RAG] BM25 索引恢复失败（退化为纯向量检索）: {e}")

    embedder = _get_embedder()
    import numpy as _np
    query_vec = embedder.encode([query])
    if isinstance(query_vec, _np.ndarray):
        norm = _np.linalg.norm(query_vec[0])
        if norm > 0:
            query_vec = query_vec / norm
        query_vec = query_vec[0].tolist()
    else:
        query_vec = query_vec[0].tolist()

    # RRF（Reciprocal Rank Fusion）参数
    RRF_K = 60  # 标准平滑常数

    def _hybrid_search(col_name: str, top_k: int, with_article: bool = True) -> list:
        """对单个 collection 跑 Dense + Sparse 并用 RRF 融合"""
        if top_k <= 0:
            return []

        # 召回窗口放大 3 倍，给 RRF 融合留余量
        recall_k = max(top_k * 3, top_k + 5)

        # ---- Dense 检索（Chroma 向量）----
        dense_hits = []
        try:
            col = client.get_collection(col_name)
            if col.count() > 0:
                res = col.query(
                    query_embeddings=[query_vec],
                    n_results=min(recall_k, col.count()),
                )
                for i, doc in enumerate(res["documents"][0]):
                    meta = res["metadatas"][0][i]
                    dist = res["distances"][0][i]
                    dense_hits.append({
                        "content": doc,
                        "source": meta.get("source", ""),
                        "chapter": meta.get("chapter", ""),
                        "article": meta.get("article", ""),
                        "layer": meta.get("layer", ""),
                        "score_dense": round(1 - dist, 4),  # cosine dist → similarity
                    })
        except Exception as e:
            print(f"[RAG] {col_name} Dense 检索失败: {e}")

        # ---- Sparse 检索（BM25）----
        sparse_hits = []
        idx = _BM25_INDEXES.get(col_name)
        if idx is not None and idx.N > 0:
            raw = idx.search(query, top_k=recall_k)
            max_bm25 = max((h.get("score", 0) for h in raw), default=1) or 1
            for h in raw:
                h["score_sparse"] = h.pop("score", 0.0)
                h["score_sparse_norm"] = round(h["score_sparse"] / max_bm25, 4)
            sparse_hits = raw

        # ---- RRF 融合（key = content 前 200 字符去重）----
        fused: dict = {}
        for rank, h in enumerate(dense_hits):
            k = h["content"][:200]
            if k not in fused:
                fused[k] = {
                    "content": h["content"],
                    "source": h.get("source", ""),
                    "chapter": h.get("chapter", ""),
                    "article": h.get("article", ""),
                    "layer": h.get("layer", ""),
                    "score_dense": h.get("score_dense", 0.0),
                    "score_sparse": 0.0,
                    "score_sparse_norm": 0.0,
                    "rank_dense": rank,
                    "rank_sparse": None,
                }
        for rank, h in enumerate(sparse_hits):
            k = h["content"][:200]
            if k not in fused:
                fused[k] = {
                    "content": h.get("content", ""),
                    "source": h.get("source", ""),
                    "chapter": h.get("chapter", ""),
                    "article": h.get("article", ""),
                    "layer": h.get("layer", ""),
                    "score_dense": 0.0,
                    "score_sparse": h.get("score_sparse", 0.0),
                    "score_sparse_norm": h.get("score_sparse_norm", 0.0),
                    "rank_dense": None,
                    "rank_sparse": rank,
                }
            else:
                # 已被向量召回，补全 BM25 排名
                fused[k]["rank_sparse"] = rank
                fused[k]["score_sparse"] = h.get("score_sparse", 0.0)
                fused[k]["score_sparse_norm"] = h.get("score_sparse_norm", 0.0)

        # 计算 RRF 分数；score 字段保留向量相似度（用户友好展示）
        for item in fused.values():
            rrf = 0.0
            if item.get("rank_dense") is not None:
                rrf += 1.0 / (RRF_K + item["rank_dense"] + 1)
            if item.get("rank_sparse") is not None:
                rrf += 1.0 / (RRF_K + item["rank_sparse"] + 1)
            item["rrf_score"] = round(rrf, 4)
            # 展示分：优先向量相似度，否则用 BM25 归一化分
            if item["score_dense"] > 0:
                item["score"] = item["score_dense"]
            else:
                item["score"] = item["score_sparse_norm"]

        # 按 RRF 降序取 top_k
        results = sorted(fused.values(), key=lambda x: -x["rrf_score"])[:top_k]
        # 清理临时字段
        for r in results:
            r.pop("rank_dense", None)
            r.pop("rank_sparse", None)
            if not with_article:
                r.pop("article", None)
        return results

    return {
        "regulation_chunks": _hybrid_search("regulation", top_k_regulation),
        "policy_chunks": _hybrid_search("policy_regulation", top_k_policy),
        "case_chunks": _hybrid_search("case_reference", top_k_case, with_article=False),
    }


def _retrieve_bm25_only(query: str, top_k_regulation: int, top_k_policy: int, top_k_case: int) -> dict:
    """纯 BM25 检索（chromadb / FlagEmbedding 不可用时的降级路径）"""
    def _search(col_name: str, top_k: int, with_article: bool = True) -> list:
        if top_k <= 0:
            return []
        idx = _BM25_INDEXES.get(col_name)
        if idx is None or idx.N == 0:
            return []
        raw = idx.search(query, top_k=top_k)
        max_bm25 = max((h.get("score", 0) for h in raw), default=1) or 1
        results = []
        for h in raw:
            h["score"] = round(h.pop("score", 0.0) / max_bm25, 4)
            h["score_sparse"] = h["score"]
            h["score_dense"] = 0.0
            h["rrf_score"] = h["score"]
            if not with_article:
                h.pop("article", None)
            results.append(h)
        return results

    return {
        "regulation_chunks": _search("regulation", top_k_regulation),
        "policy_chunks": _search("policy_regulation", top_k_policy),
        "case_chunks": _search("case_reference", top_k_case, with_article=False),
    }


# ============================================================
# 结构化规则匹配（不走向量，精确查表）
# ============================================================
def match_structured_rules(risk_points: list, rule_issues: list) -> list:
    """
    用风险点和规则问题精确匹配制度条款。
    :return: [{rule_key, matched_article, source, level, description}]
    """
    matches = []

    # 风险类型 → 制度条款关键词映射
    rule_mapping = {
        "id_card_expired": {"keywords": ["身份证过期", "有效期"], "source": "01_个人信用消费贷款准入规则.md", "article": "第一条"},
        "id_card_missing": {"keywords": ["身份证", "缺失"], "source": "01_个人信用消费贷款准入规则.md", "article": "第二条"},
        "id_mismatch": {"keywords": ["身份证号", "不一致"], "source": "04_信贷审批负面客户认定标准.md", "article": "第一条"},
        "name_mismatch": {"keywords": ["姓名", "不一致"], "source": "04_信贷审批负面客户认定标准.md", "article": "第一条"},
        "income_mismatch": {"keywords": ["收入", "不一致", "虚高"], "source": "04_信贷审批负面客户认定标准.md", "article": "第六条"},
        "cash_flow_anomaly": {"keywords": ["流水", "造假", "断裂"], "source": "04_信贷审批负面客户认定标准.md", "article": "第七条"},
        "balance_continuity": {"keywords": ["余额", "断裂"], "source": "01_个人信用消费贷款准入规则.md", "article": "第十一条"},
        "material_contradiction": {"keywords": ["材料", "矛盾"], "source": "04_信贷审批负面客户认定标准.md", "article": "第十条"},
        "suspected_fraud": {"keywords": ["造假", "伪造"], "source": "04_信贷审批负面客户认定标准.md", "article": "第九条"},
        "business_trace_missing": {"keywords": ["经营痕迹", "缺失"], "source": "02_企业经营性贷款审批管理办法.md", "article": "第九条"},
        "license_expired": {"keywords": ["证件", "过期"], "source": "02_企业经营性贷款审批管理办法.md", "article": "第五条"},
        "accounting_equation": {"keywords": ["会计恒等式"], "source": "02_企业经营性贷款审批管理办法.md", "article": "第七条"},
        "credit_report_stale": {"keywords": ["征信报告", "新鲜度"], "source": "01_个人信用消费贷款准入规则.md", "article": "第四条"},
        "key_field_missing": {"keywords": ["关键字段", "缺失"], "source": "01_个人信用消费贷款准入规则.md", "article": "第七条"},
    }

    # 从风险点提取类型
    for rp in risk_points:
        rtype = rp.get("type", "") if isinstance(rp, dict) else str(rp)
        if rtype in rule_mapping:
            m = rule_mapping[rtype]
            matches.append({
                "rule_key": rtype,
                "matched_source": m["source"],
                "matched_article": m["article"],
                "level": rp.get("level", "") if isinstance(rp, dict) else "",
                "description": rp.get("detail") or rp.get("reason", "") if isinstance(rp, dict) else "",
            })

    # 从规则问题提取
    for issue in rule_issues:
        rule_name = issue.get("rule", "")
        level = issue.get("level", "")
        msg = issue.get("msg", "")

        # 按规则名匹配
        if rule_name in rule_mapping:
            m = rule_mapping[rule_name]
            matches.append({
                "rule_key": rule_name,
                "matched_source": m["source"],
                "matched_article": m["article"],
                "level": level,
                "description": msg,
            })
        # 按关键词兜底匹配
        else:
            for key, m in rule_mapping.items():
                if any(kw in msg for kw in m["keywords"]):
                    matches.append({
                        "rule_key": key,
                        "matched_source": m["source"],
                        "matched_article": m["article"],
                        "level": level,
                        "description": msg,
                    })
                    break

    return matches


# ============================================================
# 主入口
# ============================================================
def enhance_decision_with_rag(
    risk_points: list,
    rule_issues: list,
    cross_doc_issues: list,
    missing_result: dict,
    loan_product: str,
    client=None,
) -> dict:
    """
    用 RAG 增强决策依据（三层知识库）
    :return: {
        "regulation_references": [...],  # 法规条款（最高权重，上位法依据）
        "policy_references": [...],      # 制度条款（核心依据）
        "case_references": [...],        # 参考案例
        "structured_matches": [...],     # 结构化精确匹配
    }
    """
    # 1. 结构化精确匹配（不走向量）
    structured_matches = match_structured_rules(risk_points, rule_issues)

    # 2. 向量检索法规 + 制度条款（用风险点描述作为 query）
    regulation_refs = []
    policy_refs = []
    for rp in risk_points:
        desc = rp.get("detail") or rp.get("reason", "") if isinstance(rp, dict) else str(rp)
        if not desc:
            continue
        results = retrieve(
            desc, client=client,
            top_k_regulation=2, top_k_policy=3, top_k_case=0,
        )
        for chunk in results["regulation_chunks"]:
            if not any(p["source"] == chunk["source"] and p["article"] == chunk["article"] for p in regulation_refs):
                regulation_refs.append({
                    "source": chunk["source"],
                    "article": chunk["article"],
                    "content": chunk["content"][:300],
                    "score": chunk["score"],
                })
        for chunk in results["policy_chunks"]:
            if not any(p["source"] == chunk["source"] and p["article"] == chunk["article"] for p in policy_refs):
                policy_refs.append({
                    "source": chunk["source"],
                    "article": chunk["article"],
                    "content": chunk["content"][:300],
                    "score": chunk["score"],
                })

    # 3. 用缺失项检索相关法规 + 制度
    for missing in missing_result.get("required_missing", []):
        label = missing.get("label", "")
        if label:
            results = retrieve(
                f"材料缺失 {label}", client=client,
                top_k_regulation=1, top_k_policy=2, top_k_case=0,
            )
            for chunk in results["regulation_chunks"]:
                if not any(p["source"] == chunk["source"] and p["article"] == chunk["article"] for p in regulation_refs):
                    regulation_refs.append({
                        "source": chunk["source"],
                        "article": chunk["article"],
                        "content": chunk["content"][:300],
                        "score": chunk["score"],
                    })
            for chunk in results["policy_chunks"]:
                if not any(p["source"] == chunk["source"] and p["article"] == chunk["article"] for p in policy_refs):
                    policy_refs.append({
                        "source": chunk["source"],
                        "article": chunk["article"],
                        "content": chunk["content"][:300],
                        "score": chunk["score"],
                    })

    # 4. 检索参考案例（用整体风险描述）
    risk_summary = " ".join(
        rp.get("type", "") for rp in risk_points if isinstance(rp, dict)
    )
    case_refs = []
    if risk_summary:
        results = retrieve(risk_summary, client=client, top_k_regulation=0, top_k_policy=0, top_k_case=3)
        case_refs = [{
            "source": c["source"],
            "chapter": c["chapter"],
            "content": c["content"][:400],
            "score": c["score"],
        } for c in results["case_chunks"]]

    return {
        "regulation_references": regulation_refs[:5],
        "policy_references": policy_refs[:8],
        "case_references": case_refs,
        "structured_matches": structured_matches,
    }


if __name__ == "__main__":
    # 构建知识库
    build_knowledge_base()

    # 测试检索
    print("\n" + "=" * 60)
    print("测试三层检索")
    print("=" * 60)

    test_query = "身份证号跨材料不一致，收入证明与完税证明差异大"
    print(f"\n查询: {test_query}")
    results = retrieve(test_query, top_k_regulation=2, top_k_policy=3, top_k_case=2)

    print("\n--- 法规条款（最高权重）---")
    for p in results["regulation_chunks"]:
        print(f"  [{p['score']}] {p['source']} - {p.get('article', '')}")
        print(f"    {p['content'][:100]}...")

    print("\n--- 制度条款 ---")
    for p in results["policy_chunks"]:
        print(f"  [{p['score']}] {p['source']} - {p.get('article', '')}")
        print(f"    {p['content'][:100]}...")

    print("\n--- 参考案例 ---")
    for c in results["case_chunks"]:
        print(f"  [{c['score']}] {c['source']} - {c['chapter']}")
        print(f"    {c['content'][:100]}...")
