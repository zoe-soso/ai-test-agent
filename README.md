# 基于 LLM 的 Web 智能测试用例生成与缺陷分析 Agent

> 与 `ecommerce-test-automation`（pytest + Playwright + POM）配套使用：本项目是它的"智能测试副驾"，
> 负责从需求生成用例与代码、执行并分析失败，并**对它只读，不修改其中任何文件**。

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

## 进度概览

项目按能力拆成以下模块，均已实现并通过测试：

- **需求理解与用例生成** — 读取自然语言需求，输出结构化的测试用例（覆盖正常流程、边界值、异常场景），含用例自动评审与测试数据生成。
- **代码生成（Playwright + POM）** — 把测试用例转成可执行的 Playwright 代码；生成前做静态检查，校验页面对象方法是否存在，防止大模型"编造"不存在的方法。
- **执行与人工确认** — 采用 Human-in-the-loop：AI 生成的代码在执行前必须人工确认；通过跨环境调用被测项目的 pytest 来实际跑测试。
- **失败分析与缺陷定位** — 收集失败截图与日志，由 AI 判定缺陷类别（6 类白名单）与严重度（P0–P3），并区分"偶发失败"与"稳定缺陷"，避免把网络抖动误报成 bug。
- **全链路流水线** — 一条命令把"需求 → 用例 → 代码 → 执行 → 分析"跑完，产出汇总报告。
- **多项目适配** — 通过 profile 档案接入任意 Playwright + POM 项目，换被测网站无需改代码（见下文）。
- **评测体系** — 三套可复现指标：用例覆盖率、代码首版通过率、缺陷分析准确率。

## 适配不同被测项目

"接哪个被测项目"由 `config/profiles/<名字>.yaml` 一份档案决定，不再硬编码。换被测网站只需复制
一份 yaml 改两行再 `--profile 名字` 指定，代码一行不动；页面类甚至能自动扫描发现。

```bash
python main.py run --profile ecommerce --feature "用户登录功能" --auto
python main.py run --profile myproject --feature "登录功能"   # 接新项目
```

## 评测体系

三套评测都不依赖浏览器，可每天跑、可复现：

```bash
python eval/eval_coverage.py          # 用例场景覆盖率
python eval/eval_code_passrate.py     # 代码首版 / 修正后通过率
python eval/eval_defect_accuracy.py   # 缺陷分析准确率（含人工标注集）
```

## 测试

```bash
python -m pytest     # 107 个单元测试，全绿
```
