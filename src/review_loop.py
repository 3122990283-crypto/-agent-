"""
人工审核闭环（review_loop）

让审核员能够：
1. 查看 AI 发现的问题列表（RuleFinding）
2. 逐条修改判断：confirmed（确认）/ false_positive（误报）/ escalated（升级）
3. 反馈误报原因
4. 添加漏报补充
5. 最终审核决策：approve / reject / request_more_docs / escalate

反馈数据落盘，用于后续优化规则权重（持续学习的数据基础）。

数据存储：output/review_records/{task_id}.json
"""
import os
import json
import time
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW_DIR = os.path.join(BASE_DIR, "output", "review_records")
os.makedirs(REVIEW_DIR, exist_ok=True)


# ============================================================
# 审核记录管理
# ============================================================
def save_review(task_id: str, review_data: dict) -> dict:
    """
    保存审核员的审核反馈

    :param task_id: 预审任务 ID
    :param review_data: {
        "reviewer": str,                    # 审核员
        "finding_reviews": [                # 逐条问题审核
            {
                "rule_id": str,
                "decision": "confirmed"|"false_positive"|"escalated",
                "note": str,                # 审核意见/误报原因
            }
        ],
        "missed_issues": [str],             # 审核员发现的漏报问题
        "final_decision": "approve"|"reject"|"request_more_docs"|"escalate",
        "final_note": str,                  # 总体审核意见
    }
    :return: 保存的完整记录
    """
    record = {
        "task_id": task_id,
        "reviewer": review_data.get("reviewer", "unknown"),
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finding_reviews": review_data.get("finding_reviews", []),
        "missed_issues": review_data.get("missed_issues", []),
        "final_decision": review_data.get("final_decision", ""),
        "final_note": review_data.get("final_note", ""),
    }
    path = os.path.join(REVIEW_DIR, f"{task_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def load_review(task_id: str) -> Optional[dict]:
    """加载审核记录"""
    path = os.path.join(REVIEW_DIR, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_reviews(limit: int = 50) -> list:
    """列出所有审核记录"""
    items = []
    if not os.path.exists(REVIEW_DIR):
        return items
    for fname in sorted(os.listdir(REVIEW_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(REVIEW_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
            items.append({
                "task_id": r.get("task_id"),
                "reviewer": r.get("reviewer"),
                "reviewed_at": r.get("reviewed_at"),
                "final_decision": r.get("final_decision"),
                "finding_count": len(r.get("finding_reviews", [])),
                "false_positive_count": sum(1 for fr in r.get("finding_reviews", [])
                                            if fr.get("decision") == "false_positive"),
            })
            if len(items) >= limit:
                break
        except (json.JSONDecodeError, IOError):
            continue
    return items


# ============================================================
# 规则权重优化建议（持续学习）
# ============================================================
def get_rule_feedback_stats() -> dict:
    """
    统计各规则的反馈情况，用于识别误报率高的规则
    返回 {rule_id: {confirmed: n, false_positive: n, false_positive_rate: float}}
    """
    stats = {}
    if not os.path.exists(REVIEW_DIR):
        return stats
    for fname in os.listdir(REVIEW_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(REVIEW_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
            for fr in r.get("finding_reviews", []):
                rid = fr.get("rule_id", "")
                if rid not in stats:
                    stats[rid] = {"confirmed": 0, "false_positive": 0, "escalated": 0}
                decision = fr.get("decision", "")
                if decision in stats[rid]:
                    stats[rid][decision] += 1
        except (json.JSONDecodeError, IOError):
            continue
    # 计算误报率
    for rid, s in stats.items():
        total = s["confirmed"] + s["false_positive"] + s["escalated"]
        s["total"] = total
        s["false_positive_rate"] = round(s["false_positive"] / total, 2) if total > 0 else 0
    return stats


def get_weight_adjustment_suggestions() -> list:
    """
    基于反馈统计生成规则权重调整建议
    误报率 > 30% 的规则建议降权
    """
    stats = get_rule_feedback_stats()
    suggestions = []
    for rid, s in stats.items():
        if s["total"] >= 3 and s["false_positive_rate"] > 0.3:
            suggestions.append({
                "rule_id": rid,
                "false_positive_rate": s["false_positive_rate"],
                "total_reviews": s["total"],
                "suggestion": f"规则 {rid} 误报率 {s['false_positive_rate']*100:.0f}%（{s['total']} 次审核），建议降低扣分权重或优化判定条件",
            })
    return suggestions


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    # 模拟保存审核
    tid = "test_task_001"
    review = {
        "reviewer": "审核员A",
        "finding_reviews": [
            {"rule_id": "L2-02", "decision": "confirmed", "note": "身份证确实不一致"},
            {"rule_id": "L3-05", "decision": "false_positive", "note": "USDT是客户名不是币种"},
        ],
        "missed_issues": ["收入证明章模糊"],
        "final_decision": "reject",
        "final_note": "身份主体混乱，拒贷",
    }
    saved = save_review(tid, review)
    print(f"已保存审核: {saved['task_id']} → {saved['final_decision']}")

    # 加载
    loaded = load_review(tid)
    print(f"加载: {loaded['finding_reviews']}")

    # 统计
    print(f"\n反馈统计: {get_rule_feedback_stats()}")
    print(f"权重建议: {get_weight_adjustment_suggestions()}")

    # 清理测试数据
    os.remove(os.path.join(REVIEW_DIR, f"{tid}.json"))
    print("\n测试数据已清理")
