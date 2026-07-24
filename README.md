# 信贷材料预审 Agent

基于多模态大模型 + 分层规则引擎 + 知识图谱的智能信贷材料预审系统。

## 快速启动

### 方式一：双击启动（Windows）

双击 `start.bat`，自动安装依赖并启动服务。

### 方式二：命令行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. （可选）配置 API Key
copy .env.example .env
# 编辑 .env 填入你的 OCR/LLM API Key
# 无 API Key 时本地规则引擎可正常使用

# 3. 冒烟测试
python run.py smoke

# 4. 启动 Web 服务
python run.py web
```

启动后打开浏览器访问：**http://127.0.0.1:8765/portal**

## 使用方式

1. 在上传门户输入申请人姓名
2. 上传材料文件夹（支持身份证、流水、收入证明、营业执照、不动产证等）
3. 系统自动识别贷款产品类型
4. 实时查看审核进度
5. 查看预审报告：风险评分、缺失材料、风险点、反欺诈告警

## 项目结构

```
├── run.py              # 运行入口
├── start.bat           # Windows 一键启动
├── requirements.txt    # Python 依赖
├── .env.example        # 配置模板
│
├── src/                # 核心代码
│   ├── config.py               # 配置（API Key 从 .env 读取）
│   ├── api_clients.py          # OCR / LLM API 客户端
│   ├── preaudit_agent.py       # Agent 核心调度
│   ├── business_entry.py       # 贷款产品自动识别
│   ├── layered_rules.py        # L1-L4 分层规则引擎
│   ├── risk_scoring.py         # 风险评分
│   ├── review_loop.py          # 人工审核闭环
│   ├── web_server.py           # Web 后端（FastAPI）
│   ├── step2_classify_extract.py  # OCR + 分类 + 字段抽取
│   ├── step7_inference_risk.py    # LLM 语义风险推理
│   ├── step10_rag_decision.py     # Hybrid RAG 制度检索
│   └── step11_kg_fraud.py         # 知识图谱反欺诈
│
├── knowledge_base/     # 知识库
│   ├── policies/       # 内部制度（5份）
│   ├── regulations/    # 监管法规（6份）
│   └── chroma_db/      # 向量库（运行时生成）
│
├── models/             # ML 模型
│   └── blur_classifier.joblib
│
├── static/             # Web 前端
│   ├── index.html      # 管理控制台
│   └── portal.html     # 上传门户
│
├── output/             # 输出报告（运行时生成）
└── uploads/            # 上传文件临时目录（运行时生成）
```

## API 接口

启动后可调用 REST API：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传材料并触发预审 |
| `/api/task/{task_id}` | GET | 查询任务进度/结果 |
| `/api/overview` | GET | 总览统计 |
| `/api/kg/alerts` | GET | KG 反欺诈告警 |
| `/api/rag/search` | POST | RAG 制度检索 |

## 支持的贷款产品

- 个人消费贷
- 企业经营贷
- 个人住房贷款

系统自动根据上传材料识别贷款类型，无需手动选择。

## 合规说明

本系统输出预审"建议"，高风险案例需人工复核，不做最终审批决策。
