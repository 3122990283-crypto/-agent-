"""
信贷材料预审 Agent · 配置文件
支持环境变量覆盖，可通过 .env 文件或系统环境变量配置
"""
import os

# ========== 尝试加载 .env 文件（可选，位于项目根目录） ==========
try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_ENV_PATH)
except ImportError:
    pass

# ========== API 配置 ==========

# 飞桨 PaddleOCR-VL 1.6
PADDLE_OCR_JOB_URL = os.getenv("PADDLE_OCR_JOB_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")
PADDLE_OCR_TOKEN = os.getenv("PADDLE_OCR_TOKEN", "")
PADDLE_OCR_MODEL = os.getenv("PADDLE_OCR_MODEL", "PaddleOCR-VL-1.6")

# 千问 LLM（DashScope 协议）——三模型降级，额度用完自动换下一个
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODELS = [
    os.getenv("QWEN_MODEL_1", "qwen-plus-2025-01-25"),
    os.getenv("QWEN_MODEL_2", "qwen-plus-2025-07-14"),
    os.getenv("QWEN_MODEL_3", "qwen3.6-plus"),
    os.getenv("QWEN_MODEL_4", "qwen-plus-0112"),
    os.getenv("QWEN_MODEL_5", "qwen-plus-2025-09-11"),
]
QWEN_MODEL = QWEN_MODELS[0]
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ========== 业务配置 ==========

# 支持的贷款产品（用于缺失项检测）
LOAN_PRODUCTS = {
    "personal_consumer": {
        "name": "个人消费贷",
        "required_docs": [
            {"doc_type": "id_card_front", "required": True, "label": "身份证正面"},
            {"doc_type": "id_card_back", "required": True, "label": "身份证反面"},
            {"doc_type": "bank_statement", "required": True, "label": "银行流水（近6个月）"},
            {"doc_type": "income_certificate", "required": True, "label": "收入证明"},
            {"doc_type": "loan_investigation", "required": True, "label": "个人贷款调查报告"},
        ],
    },
    "business_loan": {
        "name": "企业经营贷",
        "required_docs": [
            {"doc_type": "business_license", "required": True, "label": "营业执照"},
            {"doc_type": "id_card_front", "required": True, "label": "法人身份证正面"},
            {"doc_type": "id_card_back", "required": True, "label": "法人身份证反面"},
            {"doc_type": "bank_statement", "required": True, "label": "企业银行流水（近12个月）"},
            {"doc_type": "fund_flow_analysis", "required": True, "label": "资金流水分析表"},
            {"doc_type": "loan_investigation", "required": True, "label": "个人贷款调查报告"},
        ],
    },
    "mortgage": {
        "name": "个人住房贷款",
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
    },
}

# ========== 风控阈值 ==========

RISK_THRESHOLDS = {
    "dti_warning": float(os.getenv("DTI_WARNING", "0.5")),
    "income_mismatch": float(os.getenv("INCOME_MISMATCH", "0.2")),
    "credit_report_fresh_days": int(os.getenv("CREDIT_REPORT_FRESH_DAYS", "15")),
    "overdue_total_threshold": int(os.getenv("OVERDUE_TOTAL_THRESHOLD", "6")),
    "overdue_consecutive_threshold": int(os.getenv("OVERDUE_CONSECUTIVE_THRESHOLD", "3")),
    "flow_date_gap_days": int(os.getenv("FLOW_DATE_GAP_DAYS", "30")),
}

# ========== 图像质量阈值 ==========

IMAGE_QUALITY = {
    "blur_variance_threshold": 40,
    "tenengrad_threshold": 1000,
    "blur_score_threshold": 0.35,
    "ml_blur_threshold": 0.95,
    "clipiqa_threshold": 0.30,
    "too_bright_threshold": 248,
    "too_dark_threshold": 30,
    "noise_block_ratio": 0.5,
    "noise_var_high": 8000,
    "noise_var_low": 5,
    "border_required": False,
    "min_image_size": 200,
    "max_file_size_mb": 20,
    "max_pdf_pages": 50,
    "multi_doc_area_ratio": 0.15,
    "tilt_angle_threshold": 15.0,
}

# ========== 置信度权重 ==========

CONFIDENCE_WEIGHTS = {
    "quality": 0.15,
    "ocr": 0.25,
    "rule": 0.30,
    "inference": 0.30,
}

# 自动通过/转人工阈值
AUTO_PASS_THRESHOLD = float(os.getenv("AUTO_PASS_THRESHOLD", "0.9"))
MANUAL_REVIEW_THRESHOLD = float(os.getenv("MANUAL_REVIEW_THRESHOLD", "0.6"))

# ========== 数据集路径（相对路径，自动定位） ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.getenv("DATASET_ROOT", os.path.join(BASE_DIR, "数据集"))
