# 基于 LLM 的 Web 智能测试用例生成与缺陷分析 Agent

> 30 天计划中的第二个项目。第一个项目 `ecommerce-test-automation`
> （pytest + Playwright + POM）是本项目的数据来源与执行环境，
> **本项目对它只读，不会修改其中任何文件**。

## 这个项目要做什么

```
自然语言需求  →  AI 分析  →  结构化测试用例  →  Playwright 代码
      →  人工确认  →  pytest 执行  →  截图/日志  →  AI 失败分析  →  测试报告
```

采用 **Human-in-the-loop**：AI 生成的代码在执行前必须人工确认。
这不是"偷懒"，而是生产环境的基本要求——不能让 Agent 随便执行它自己写的代码。

## 环境隔离说明（重要）

| 项目 | 虚拟环境 | 主要依赖 |
|---|---|---|
| `ecommerce-test-automation` | 它自己的 `venv/`（Python 3.12.1） | playwright, pytest, allure-pytest, PyYAML, faker |
| 本项目 `ai-test-agent` | 自己的 `venv/`（Python 3.12.1） | openai, python-dotenv, PyYAML, pytest |

两个项目各自独立建环境，原因：

1. 依赖互不污染。以后接 LangGraph、重试库等，不会影响已能跑通的测试项目。
2. 两份 `requirements.txt` 各管各的，交付和复现都干净。
3. Agent 需要跑 pytest 时，用**绝对路径**跨环境调用测试项目的解释器：
   `D:/PythonProjects/ecommerce-test-automation/venv/Scripts/python.exe -m pytest`
   这是很常见的跨环境编排方式，也是面试能讲的设计点。

Python 版本刻意选成和测试项目一致的 3.12.1，避免两边行为不一致。

## 快速开始

```bash
# 1. 建虚拟环境（Windows）
python -m venv venv

# 2. 激活
venv\Scripts\activate

# 3. 装依赖
pip install -r requirements.txt

# 4. 配置密钥
cp .env.example .env     # 然后填入 LLM_API_KEY

# 5. 自检
python main.py
```

## 目录结构

```
ai-test-agent/
├── agent/          # Agent 核心：LLM 客户端、用例生成、用例构建
├── tools/          # 工具层：日志、文件读写、异常定义
├── prompts/        # Prompt 模板（.txt，和代码分离，改文案不用改代码）
├── config/         # 全局配置与路径
├── data/           # 输入：需求文档
├── outputs/        # 产出：测试用例 YAML、报告
├── tests/          # 本项目自身的单元测试
├── logs/           # 运行日志
├── main.py         # 命令行入口
└── requirements.txt
```

## 进度

| 天数 | 主题 | 状态 |
|---|---|---|
| Day 1 | Python 环境与项目准备 | ✅ |
| Day 2 | 函数 + 数据结构，纯 Python 用例生成器 | ✅ |
| Day 3 | JSON / YAML / 文件操作 | ✅ |
| Day 4 | 异常处理 + 日志 | ✅ |
| Day 5 | 第一次调用大模型 | ✅ |

详细笔记见 `../docs/notes/`，每日日志见 `../docs/daily/`。
