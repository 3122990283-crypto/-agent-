"""
信贷材料预审 Agent · Web 控制台后端

提供 REST API：
- GET  /api/overview           总览统计
- GET  /api/applicants         申请人列表
- GET  /api/applicant/{name}   申请人详情
- GET  /api/kg                 知识图谱（聚合后）
- GET  /api/kg/alerts          KG 反欺诈告警
- POST /api/rag/search         Hybrid RAG 检索
- GET  /api/eval               RAG 评估指标汇总
- POST /api/upload             上传材料 + 触发预审（异步）
- GET  /api/task/{task_id}     查询预审任务进度/结果
- GET  /api/products           贷款产品列表
- GET  /api/history            上传历史

启动：
    python web_server.py
    # 控制台   http://127.0.0.1:8765
    # 上传门户  http://127.0.0.1:8765/portal
"""
import os
import json
import asyncio
import uuid
import time
import shutil
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="信贷预审 Agent 控制台", version="1.0")


# ============================================================
# 异步任务存储（内存 + 落盘）
# ============================================================
_TASKS: dict = {}  # {task_id: {status, progress, step, result, error, created_at, ...}}
_TASKS_LOCK = threading.Lock()
_TASKS_FILE = os.path.join(OUTPUT_DIR, "upload_tasks.json")


def _persist_tasks():
    """持久化任务历史到磁盘"""
    try:
        snapshot = {}
        with _TASKS_LOCK:
            for tid, t in _TASKS.items():
                # 不持久化大字段 result（可能很大）
                snapshot[tid] = {
                    k: v for k, v in t.items()
                    if k in ("status", "progress", "step", "created_at", "completed_at",
                            "person", "loan_product", "file_count", "error", "decision", "summary",
                            "product_info")
                }
        with open(_TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_tasks():
    """从磁盘恢复任务历史"""
    try:
        if os.path.exists(_TASKS_FILE):
            with open(_TASKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with _TASKS_LOCK:
                for tid, t in data.items():
                    t.setdefault("status", "done")
                    t.setdefault("progress", 100)
                    _TASKS[tid] = t
    except Exception:
        pass


_load_tasks()


# ============================================================
# 数据加载（懒加载 + 缓存）
# ============================================================
_report_cache: Optional[dict] = None
_kg_cache: Optional[dict] = None
_kg_alerts_cache: Optional[list] = None


def _load_report() -> dict:
    global _report_cache
    if _report_cache is None:
        path = os.path.join(OUTPUT_DIR, "demo_batch_report.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="demo_batch_report.json 未生成，请先运行 demo_batch.py")
        with open(path, "r", encoding="utf-8") as f:
            _report_cache = json.load(f)
    return _report_cache


def _load_kg() -> dict:
    global _kg_cache
    if _kg_cache is None:
        path = os.path.join(OUTPUT_DIR, "kg_graph.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="kg_graph.json 未生成")
        with open(path, "r", encoding="utf-8") as f:
            full = json.load(f)
        # 聚合：剔除 Document 节点（310 个，太密），只保留实体节点 + Applicant
        # 同时剔除 SUBMITTED 边（申请人→文档），只保留实体关系边
        keep_types = {"Applicant", "IDNumber", "Phone", "Address", "Employer", "BankAccount"}
        keep_ids = set()
        slim_nodes = []
        for n in full.get("nodes", []):
            if n.get("node_type") in keep_types:
                slim_nodes.append(n)
                keep_ids.add(n["id"])
        slim_edges = []
        for e in full.get("edges", []):
            if e.get("relation") == "SUBMITTED":
                continue
            if e.get("src") in keep_ids and e.get("dst") in keep_ids:
                slim_edges.append(e)
        _kg_cache = {
            "nodes": slim_nodes,
            "edges": slim_edges,
            "stats": {
                "node_total": len(slim_nodes),
                "edge_total": len(slim_edges),
                "full_node_total": len(full.get("nodes", [])),
                "full_edge_total": len(full.get("edges", [])),
            },
        }
    return _kg_cache


def _load_kg_alerts() -> list:
    global _kg_alerts_cache
    if _kg_alerts_cache is None:
        path = os.path.join(OUTPUT_DIR, "kg_fraud_report.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="kg_fraud_report.json 未生成")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _kg_alerts_cache = data.get("fraud_alerts", [])
    return _kg_alerts_cache


# ============================================================
# API
# ============================================================
@app.get("/api/overview")
def api_overview():
    """总览统计"""
    rep = _load_report()
    results = rep.get("results", [])
    decision_dist = rep.get("decision_distribution", {})
    risk_dist = {}
    confidence_list = []
    missing_counter = {}
    risk_type_counter = {}
    for r in results:
        risk_dist[r.get("risk_level", "unknown")] = risk_dist.get(r.get("risk_level", "unknown"), 0) + 1
        conf = r.get("confidence", {})
        confidence_list.append({
            "person": r.get("person"),
            "total": conf.get("total_confidence", 0),
            "quality": conf.get("quality_confidence", 0),
            "ocr": conf.get("ocr_confidence", 0),
            "rule": conf.get("rule_confidence", 0),
            "inference": conf.get("inference_confidence", 0),
            "completeness": conf.get("data_completeness", 0),
        })
        for m in r.get("missing_required", []) or []:
            missing_counter[m] = missing_counter.get(m, 0) + 1
        for rp in r.get("risk_points", []) or []:
            t = rp.get("type", "unknown")
            risk_type_counter[t] = risk_type_counter.get(t, 0) + 1
    # 评估指标均值
    evals = [r.get("eval_metrics") for r in results if r.get("eval_metrics")]
    eval_summary = None
    if evals:
        eval_summary = {
            "count": len(evals),
            "avg_faithfulness": round(sum(e.get("faithfulness", {}).get("faithfulness", 0) for e in evals) / len(evals), 4),
            "avg_answer_relevancy": round(sum(e.get("answer_relevancy", {}).get("answer_relevancy", 0) for e in evals) / len(evals), 4),
            "avg_overall": round(sum(e.get("overall", 0) for e in evals) / len(evals), 4),
        }
    return {
        "generated_at": rep.get("generated_at"),
        "applicant_count": rep.get("applicant_count"),
        "total_elapsed_seconds": rep.get("total_elapsed_seconds"),
        "decision_distribution": decision_dist,
        "risk_distribution": risk_dist,
        "missing_top": sorted(missing_counter.items(), key=lambda x: -x[1])[:10],
        "risk_type_distribution": risk_type_counter,
        "confidence_list": confidence_list,
        "eval_summary": eval_summary,
    }


@app.get("/api/applicants")
def api_applicants():
    """申请人列表（精简）"""
    rep = _load_report()
    items = []
    for r in rep.get("results", []):
        conf = r.get("confidence", {})
        items.append({
            "person": r.get("person"),
            "doc_count": r.get("doc_count"),
            "doc_types": r.get("doc_types", []),
            "decision": r.get("decision"),
            "risk_level": r.get("risk_level"),
            "risk_points_count": r.get("risk_points_count"),
            "missing_required_count": len(r.get("missing_required", []) or []),
            "missing_required": r.get("missing_required", []),
            "reject_reason": r.get("reject_reason", ""),
            "total_confidence": conf.get("total_confidence"),
            "elapsed_seconds": r.get("elapsed_seconds"),
        })
    return {"applicants": items}


@app.get("/api/applicant/{name}")
def api_applicant_detail(name: str):
    """申请人详情"""
    rep = _load_report()
    for r in rep.get("results", []):
        if r.get("person") == name:
            return r
    raise HTTPException(status_code=404, detail=f"申请人 {name} 不存在")


@app.get("/api/kg")
def api_kg():
    """知识图谱（聚合后）"""
    return _load_kg()


@app.get("/api/kg/alerts")
def api_kg_alerts():
    """KG 反欺诈告警"""
    return {"alerts": _load_kg_alerts()}


# ---- RAG 检索 ----
class RagQuery(BaseModel):
    query: str
    top_k_regulation: int = 3
    top_k_policy: int = 5
    top_k_case: int = 3


@app.post("/api/rag/search")
async def api_rag_search(q: RagQuery):
    """Hybrid RAG 检索（Dense BGE-M3 + Sparse BM25 RRF 融合）"""
    try:
        # 在子线程跑，避免阻塞事件循环（首次加载 BGE-M3 较慢）
        import step10_rag_decision as s
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: s.retrieve(
                q.query,
                top_k_regulation=q.top_k_regulation,
                top_k_policy=q.top_k_policy,
                top_k_case=q.top_k_case,
            )
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 检索失败: {e}")


@app.get("/api/eval")
def api_eval():
    """RAG 评估指标汇总"""
    rep = _load_report()
    items = []
    for r in rep.get("results", []):
        em = r.get("eval_metrics")
        if em:
            items.append({
                "person": r.get("person"),
                "faithfulness": em.get("faithfulness", {}).get("faithfulness"),
                "answer_relevancy": em.get("answer_relevancy", {}).get("answer_relevancy"),
                "overall": em.get("overall"),
                "unsupported_count": len(em.get("faithfulness", {}).get("unsupported", [])),
            })
    return {"items": items}


# ============================================================
# 上传 + 异步预审
# ============================================================
def _update_task(task_id: str, **kwargs):
    """更新任务状态（线程安全）"""
    with _TASKS_LOCK:
        if task_id in _TASKS:
            _TASKS[task_id].update(kwargs)
        else:
            _TASKS[task_id] = kwargs
    _persist_tasks()


def _run_preaudit(task_id: str, person: str, file_paths: list, loan_product: str):
    """在后台线程跑预审 Agent（新架构：Agent 动态调度工具集）"""
    try:
        _update_task(task_id, status="running", progress=5, step="Agent 启动 · OCR + 抽取")
        import preaudit_agent

        def _cb(progress, step, extra=None):
            update_kwargs = {"progress": progress, "step": step}
            if extra and isinstance(extra, dict):
                if "product_info" in extra:
                    update_kwargs["product_info"] = extra["product_info"]
            _update_task(task_id, **update_kwargs)

        result = preaudit_agent.run_preaudit_agent(
            person, file_paths, loan_product_hint=loan_product, progress_callback=_cb
        )

        _update_task(
            task_id,
            status="done",
            progress=100,
            step="完成 · 决策报告已生成",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            result=result,
            decision=result.get("decision"),
            summary=result.get("summary") or result.get("reject_reason", ""),
        )
    except Exception as e:
        import traceback
        _update_task(
            task_id,
            status="error",
            progress=100,
            step="失败",
            error=f"{e}",
            error_trace=traceback.format_exc(),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )


@app.get("/api/products")
def api_products():
    """贷款产品列表"""
    try:
        import business_entry
        return {"products": business_entry.list_products()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def api_upload(
    files: list[UploadFile] = File(...),
    person: str = Form(""),
    loan_product: str = Form(""),
):
    """
    上传材料 + 触发预审
    - files: 多个文件（图片/PDF）
    - person: 申请人姓名（可空，默认匿名）
    - loan_product: 贷款产品 key
    """
    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件")

    task_id = f"task_{uuid.uuid4().hex[:12]}"
    task_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    saved_paths = []
    seen_names = set()
    for idx, f in enumerate(files):
        safe_name = os.path.basename(f.filename or "unnamed")
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".pdf", ".webp", ".tiff"}:
            continue
        base = os.path.splitext(safe_name)[0]
        if safe_name in seen_names:
            safe_name = f"{base}_{idx}{ext}"
        seen_names.add(safe_name)
        dest = os.path.join(task_dir, safe_name)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="无有效文件（仅支持 jpg/png/bmp/pdf/webp/tiff）")

    person = person.strip() or f"上传申请人_{task_id[-4:]}"

    # 初始化任务
    _update_task(
        task_id,
        status="pending",
        progress=0,
        step="已接收，等待处理",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        person=person,
        loan_product=loan_product,
        file_count=len(saved_paths),
        file_names=[os.path.basename(p) for p in saved_paths],
    )

    # 后台线程跑预审
    thread = threading.Thread(
        target=_run_preaudit,
        args=(task_id, person, saved_paths, loan_product),
        daemon=True,
    )
    thread.start()

    return {
        "task_id": task_id,
        "person": person,
        "file_count": len(saved_paths),
        "message": "已提交，预审进行中",
    }


@app.get("/api/task/{task_id}")
def api_task_status(task_id: str):
    """查询任务进度/结果"""
    with _TASKS_LOCK:
        t = _TASKS.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")
    # 返回时浅拷贝避免外部修改
    return dict(t)


@app.get("/api/history")
def api_history(limit: int = Query(20, ge=1, le=100)):
    """上传历史"""
    with _TASKS_LOCK:
        items = []
        for tid, t in sorted(_TASKS.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)[:limit]:
            items.append({
                "task_id": tid,
                "person": t.get("person"),
                "loan_product": t.get("loan_product"),
                "file_count": t.get("file_count"),
                "status": t.get("status"),
                "decision": t.get("decision"),
                "summary": t.get("summary"),
                "created_at": t.get("created_at"),
                "completed_at": t.get("completed_at"),
            })
    return {"items": items, "total": len(_TASKS)}


# ============================================================
# 人工审核闭环 API
# ============================================================
@app.get("/api/reviews")
def api_list_reviews(limit: int = Query(50, ge=1, le=200)):
    """列出所有审核记录"""
    import review_loop
    return {"items": review_loop.list_reviews(limit)}


@app.get("/api/review/feedback-stats")
def api_feedback_stats():
    """规则反馈统计（用于持续学习）"""
    import review_loop
    return {
        "rule_stats": review_loop.get_rule_feedback_stats(),
        "weight_suggestions": review_loop.get_weight_adjustment_suggestions(),
    }


@app.post("/api/review/{task_id}")
def api_submit_review(task_id: str, review_data: dict):
    """提交审核员的审核反馈"""
    try:
        import review_loop
        record = review_loop.save_review(task_id, review_data)
        return {"status": "saved", "record": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/{task_id}")
def api_get_review(task_id: str):
    """获取某任务的审核记录"""
    import review_loop
    r = review_loop.load_review(task_id)
    if not r:
        raise HTTPException(status_code=404, detail="该任务暂无审核记录")
    return r


# ============================================================
# 静态资源
# ============================================================
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/portal")
def portal():
    return FileResponse(os.path.join(STATIC_DIR, "portal.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    os.makedirs(STATIC_DIR, exist_ok=True)
    print("=" * 60)
    print("  信贷材料预审 Agent · Web 服务")
    print("  控制台    http://127.0.0.1:8765")
    print("  上传门户  http://127.0.0.1:8765/portal")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
