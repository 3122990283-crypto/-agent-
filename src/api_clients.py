"""
API 客户端封装：飞桨 PaddleOCR-VL 1.6 + 千问 LLM
"""
import os
import json
import time
import base64
import requests
from openai import OpenAI

import config


# ============================================================
# 飞桨 PaddleOCR-VL 1.6 客户端（异步任务模式）
# ============================================================
class PaddleOCRClient:
    """飞桨 PaddleOCR-VL 1.6 文档解析 API"""

    def __init__(self):
        self.job_url = config.PADDLE_OCR_JOB_URL
        self.token = config.PADDLE_OCR_TOKEN
        self.model = config.PADDLE_OCR_MODEL
        self.headers = {"Authorization": f"bearer {self.token}"}

    def _submit_job(self, file_path: str) -> str:
        """提交 OCR 任务，返回 jobId"""
        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }

        if file_path.startswith("http"):
            # URL 模式
            headers = {**self.headers, "Content-Type": "application/json"}
            payload = {
                "fileUrl": file_path,
                "model": self.model,
                "optionalPayload": optional_payload,
            }
            resp = requests.post(self.job_url, json=payload, headers=headers)
        else:
            # 本地文件模式
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            data = {
                "model": self.model,
                "optionalPayload": json.dumps(optional_payload),
            }
            with open(file_path, "rb") as f:
                files = {"file": f}
                resp = requests.post(
                    self.job_url, headers=self.headers, data=data, files=files
                )

        if resp.status_code != 200:
            raise RuntimeError(f"提交任务失败 [{resp.status_code}]: {resp.text}")

        job_id = resp.json()["data"]["jobId"]
        print(f"[PaddleOCR] 任务已提交, jobId={job_id}")
        return job_id

    def _poll_result(self, job_id: str, timeout: int = 300, interval: int = 5) -> dict:
        """轮询任务结果"""
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(f"{self.job_url}/{job_id}", headers=self.headers)
            if resp.status_code != 200:
                raise RuntimeError(f"轮询失败 [{resp.status_code}]: {resp.text}")

            data = resp.json()["data"]
            state = data["state"]

            if state == "done":
                print(f"[PaddleOCR] 任务完成")
                return data
            elif state == "failed":
                raise RuntimeError(f"任务失败: {data.get('errorMsg')}")
            else:
                # pending / running
                try:
                    progress = data["extractProgress"]
                    print(
                        f"[PaddleOCR] {state}, 页数: "
                        f"{progress.get('extractedPages', 0)}/{progress.get('totalPages', '?')}"
                    )
                except KeyError:
                    print(f"[PaddleOCR] {state}...")
                time.sleep(interval)

        raise TimeoutError(f"任务超时, jobId={job_id}")

    def _download_jsonl(self, jsonl_url: str) -> list:
        """下载 jsonl 结果，返回每页的解析结果"""
        resp = requests.get(jsonl_url)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        pages = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)["result"]
            pages.append(result)
        return pages

    def parse(self, file_path: str) -> dict:
        """
        解析文档，返回结构化结果
        :return: {
            "pages": [{markdown, layoutParsingResults}],
            "raw": 原始返回
        }
        """
        job_id = self._submit_job(file_path)
        data = self._poll_result(job_id)
        jsonl_url = data["resultUrl"]["jsonUrl"]
        pages = self._download_jsonl(jsonl_url)

        # 提取每页 markdown 文本
        page_texts = []
        for page in pages:
            md_text = ""
            for res in page.get("layoutParsingResults", []):
                md_text += res.get("markdown", {}).get("text", "")
            page_texts.append(md_text)

        return {
            "pages": pages,
            "page_texts": page_texts,
            "full_text": "\n".join(page_texts),
            "job_id": job_id,
        }

    def parse_batch(self, file_paths: list, max_workers: int = 8) -> list:
        """
        批量并发解析（高吞吐）
        API 为异步任务模式，天然支持并发：同时提交多个任务，再并发轮询

        :param file_paths: 文件路径列表
        :param max_workers: 并发数（默认 8）
        :return: 与输入顺序对应的解析结果列表（失败项返回 {"error": ...}）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(file_paths)

        def _worker(idx, path):
            try:
                r = self.parse(path)
                return idx, r
            except Exception as e:
                return idx, {"error": str(e), "file": path}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_worker, i, p) for i, p in enumerate(file_paths)]
            done = 0
            for fut in as_completed(futures):
                idx, r = fut.result()
                results[idx] = r
                done += 1
                ok = "error" not in r
                print(f"[PaddleOCR] 批量进度: {done}/{len(file_paths)} "
                      f"({'✓' if ok else '✗'} {os.path.basename(file_paths[idx])})")
        return results


# ============================================================
# 千问 LLM 客户端（OpenAI 兼容协议，四模型自动降级）
# ============================================================
class QwenClient:
    """
    千问 LLM 客户端（DashScope OpenAI 兼容协议）

    支持四模型降级：额度用完（429/额度耗尽）自动切换下一个模型
    优先级：qwen3-max-2025-09-23 → qwen3-max → qwen3.7-max → qwen3.7-max-2026-06-08
    """

    # 触发降级的错误关键词（额度耗尽 / 限流 / 模型不可用）
    _FALLBACK_KEYWORDS = (
        "quota", "rate limit", "exceeded", "insufficient",
        "余额不足", "额度", "限流", "配额",
    )

    def __init__(self):
        self.client = OpenAI(
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
        )
        self.models = list(getattr(config, "QWEN_MODELS", [config.QWEN_MODEL]))
        # 当前使用的模型索引（默认从第一个开始）
        self._current_idx = 0
        # 记录每个模型的失败状态（避免反复尝试已耗尽的模型）
        self._exhausted = set()

    @property
    def model(self) -> str:
        """当前使用的模型"""
        return self.models[self._current_idx]

    def _try_next_model(self) -> bool:
        """切换到下一个未耗尽的模型，返回是否切换成功"""
        for i in range(self._current_idx + 1, len(self.models)):
            if i not in self._exhausted:
                self._current_idx = i
                print(f"[Qwen] 模型降级: → {self.models[i]}")
                return True
        return False

    def _should_fallback(self, error: Exception) -> bool:
        """判断错误是否应该触发降级"""
        err_str = str(error).lower()
        for kw in self._FALLBACK_KEYWORDS:
            if kw.lower() in err_str:
                return True
        # HTTP 429
        if hasattr(error, "status_code") and error.status_code == 429:
            return True
        # openai.RateLimitError
        if "RateLimitError" in type(error).__name__:
            return True
        return False

    def chat(self, prompt: str, system: str = None, temperature: float = 0.1) -> str:
        """
        普通对话（带自动降级）
        额度用完自动切换到下一个模型，全部耗尽则抛最后一个错误
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(len(self.models)):
            model = self.model
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"enable_thinking": False},
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_error = e
                if self._should_fallback(e):
                    print(f"[Qwen] {model} 额度耗尽/限流: {str(e)[:100]}")
                    self._exhausted.add(self._current_idx)
                    if not self._try_next_model():
                        raise RuntimeError(
                            f"所有千问模型均已耗尽: {self.models}"
                        ) from e
                    # 切换后继续尝试新模型
                    continue
                else:
                    # 非额度错误，直接抛出
                    raise

        raise RuntimeError(f"所有千问模型调用失败: {self.models}") from last_error

    def chat_json(self, prompt: str, system: str = None, temperature: float = 0.1) -> dict:
        """
        对话并解析 JSON 输出
        自动剥离 ```json ... ``` 包裹
        """
        raw = self.chat(prompt, system, temperature)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 输出中提取 JSON"""
        # 去掉 think 标签（qwen3 思考过程）
        if "</think>" in text:
            text = text.split("</think>")[-1]

        # 去掉 ```json 包裹
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()
        return json.loads(text)


# ============================================================
# 单例
# ============================================================
_paddle_client = None
_qwen_client = None


def get_paddle_ocr() -> PaddleOCRClient:
    global _paddle_client
    if _paddle_client is None:
        _paddle_client = PaddleOCRClient()
    return _paddle_client


def get_qwen() -> QwenClient:
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = QwenClient()
    return _qwen_client
