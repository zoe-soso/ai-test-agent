"""
全局配置模块（Day 1）

为什么要有这个文件？
    新手最常见的坏习惯是：把路径、模型名、超时时间这些"魔法值"
    散落在各个 .py 文件里。改一个路径要翻 5 个文件，还容易漏。
    这里集中定义，其他模块一律 `from config.settings import XXX`。

设计约定：
    1. 路径用 pathlib.Path，不要用字符串拼接（Windows 反斜杠会坑你）。
    2. 敏感信息（API Key）只从 .env 读取，绝不写死在代码里。
    3. 本模块"导入即安全"：即使没有 .env 也不会报错，只是 LLM 不可用。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. 路径
# ------------------------------------------------------------------
# __file__ 是本文件；parent 是 config/；parent.parent 是项目根 ai-test-agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"          # 放输入：需求文档
OUTPUT_DIR = PROJECT_ROOT / "outputs"     # 放产出：测试用例、报告
LOG_DIR = PROJECT_ROOT / "logs"           # 放日志
PROMPT_DIR = PROJECT_ROOT / "prompts"     # 放 Prompt 模板
EVAL_DIR = PROJECT_ROOT / "eval"          # 放评测集与评测结果（Day 7）

# ------------------------------------------------------------------
# 2. 加载 .env（敏感信息）
# ------------------------------------------------------------------
# load_dotenv 会向上查找 .env 文件并写入 os.environ。
# override=False：已存在的环境变量优先，方便 CI / 命令行临时覆盖。
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ------------------------------------------------------------------
# 3. 大模型配置（Day 5 使用）
# ------------------------------------------------------------------
# 统一用 OpenAI 兼容协议，换供应商只改 .env，不改代码。
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))       # 秒
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# 离线演练开关（Day 5 会解释为什么需要它）
#   "auto"  —— 没配 Key 就自动用 Mock，配了就走真实调用（默认，最省心）
#   "true"  —— 强制 Mock，不花钱也能反复跑通流程
#   "false" —— 强制真实调用，没 Key 就直接报错（接 CI 时用，避免静默降级）
LLM_MOCK_MODE: str = os.getenv("LLM_MOCK", "auto").lower()

# 假模型的"表演风格"（Day 7 起）。可选值见 agent/mock_llm.py：
#   cycle   按 clean -> fenced -> chatty -> broken 轮换（默认）
#           连续跑几次就能自然撞上 broken，从而验证自修正逻辑
#   clean   永远返回标准 JSON
#   fenced  永远包一层 ```json 代码围栏
#   chatty  在 JSON 前后加客套话
#   broken  故意返回缺字段 / 枚举越界的 JSON
# 写单元测试或复现某个 bug 时，把它设成固定值，比靠轮换碰运气可靠得多。
LLM_MOCK_STYLE: str = os.getenv("LLM_MOCK_STYLE", "cycle").lower()

# ------------------------------------------------------------------
# 4. 已存在的测试项目（ecommerce-test-automation）
# ------------------------------------------------------------------
# 重要：本项目对它只做"读取 + 通过子进程调用 pytest"，绝不修改其中任何文件。
# 两个项目用各自的虚拟环境，通过绝对路径互相调用，互不污染。
TEST_PROJECT_DIR = Path(
    os.getenv(
        "TEST_PROJECT_DIR",
        r"D:/PythonProjects/ecommerce-test-automation",
    )
)
# 测试项目自己的解释器。之后 Agent 要跑 pytest 时用它，而不是本项目 venv。
TEST_PROJECT_PYTHON = TEST_PROJECT_DIR / "venv" / "Scripts" / "python.exe"

# ------------------------------------------------------------------
# 5. 日志配置（Day 4 使用）
# ------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ------------------------------------------------------------------
# 6. 成本配置（Day 7 使用）
# ------------------------------------------------------------------
# LLM 项目在真实工程里受两个硬约束：**钱** 和 **延迟**。
# 面试时能说出"我把单条需求生成 8 条用例的成本控制在 X 分钱"，
# 比"我用了大模型"专业一个量级。所以从 Day 7 起就把成本算进评测。

# 单位：元 / 每 100 万 token。
# ⚠️ 各家价格随时会调整，下面的默认值只是**量级参考**，
#    不保证是你账号当前的实际单价。换供应商或调价后，
#    请在 .env 里覆盖这两个值，代码不用动。
PRICE_INPUT_PER_1M: float = float(os.getenv("PRICE_INPUT_PER_1M", "2.0"))
PRICE_OUTPUT_PER_1M: float = float(os.getenv("PRICE_OUTPUT_PER_1M", "8.0"))
CURRENCY: str = os.getenv("CURRENCY", "¥")


def ensure_dirs() -> None:
    """确保所有输出目录都存在。

    为什么要显式创建？
        往不存在的目录写文件会抛 FileNotFoundError。
        在程序启动时统一创建一次，比在每个写文件的地方 try 一遍干净得多。
    """
    for directory in (DATA_DIR, OUTPUT_DIR, LOG_DIR, PROMPT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 评测目录多两层：eval/cases 放评测集，eval/runs 放每次结果快照
    for directory in (EVAL_DIR, EVAL_DIR / "cases", EVAL_DIR / "runs"):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> dict:
    """返回一份"当前环境体检报告"，供 main.py 打印。

    返回 dict 而不是直接 print，是为了让函数保持"可测试"：
    函数负责算，调用方负责展示。这是后面写单元测试的前提。
    """
    return {
        "项目根目录": str(PROJECT_ROOT),
        "Python 版本": _python_version(),
        "LLM 供应商地址": LLM_BASE_URL,
        "LLM 模型": LLM_MODEL,
        "API Key 是否已配置": bool(LLM_API_KEY),
        "离线 Mock 策略": LLM_MOCK_MODE,
        "关联测试项目": str(TEST_PROJECT_DIR),
        "测试项目解释器是否存在": TEST_PROJECT_PYTHON.exists(),
        "测试项目是否被改动过": "否（本项目只读引用）",
    }


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


if __name__ == "__main__":
    # 直接 `python config/settings.py` 就能自检环境
    for key, value in describe().items():
        print(f"{key}: {value}")
