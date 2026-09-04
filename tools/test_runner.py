"""
pytest 执行器（Day 20）

------------------------------------------------------------------
这一步为什么是分水岭
------------------------------------------------------------------
在 Day 20 之前，我们的 Agent 只会"生成文本"：用例、数据、代码，全是文件。
从今天起，它能**真的去做一件事**——调用 pytest 跑测试，并拿到真实结果。

    生成代码 → 人工确认 → 执行 → 拿到 exit code / 输出 → 解析成结构化结果

有了"执行结果"这个输入，后面 Day 23~26 才能做失败分析。
**没有执行结果，就没有可分析的东西** —— 所以这一步是后面所有能力的前提。

------------------------------------------------------------------
用 subprocess 执行外部命令，要注意的三件事
------------------------------------------------------------------
1. **命令模板写死，只留参数口子**
   模型（或用户）只能通过 target 参数影响范围，塞不进任何 shell 语法。
   本项目固定执行 `[python, "-m", "pytest", <path>, ...]`，不拼接字符串。

2. **必须设超时**
   浏览器测试可能卡死（页面加载不出来、元素等不到）。
   没有超时，一个挂死的测试能让整个 Agent 卡一整晚。

3. **一定要拿到 exit code（退出码）**
   很多新手只看"有没有报错信息"，这是不可靠的。
   pytest 用退出码表达结果，这是最权威的判据：
       0 = 全部通过
       1 = 有用例失败
       2 = 执行被中断
       3 = 内部错误
       4 = 命令行用法错误
       5 = 没有收集到任何用例

------------------------------------------------------------------
跨项目执行的关键处理
------------------------------------------------------------------
生成的测试代码放在**我们自己的** generated_tests/ 目录，
但要借用 ecommerce-test-automation 的虚拟环境来跑
（因为 Playwright、pytest 插件、页面对象都在那边）。

做法：用对方的 python.exe，但把工作目录切到对方项目根，
并把对方目录加进 PYTHONPATH，这样 `import pages.login_page` 才能成功。
**整个过程不往对方项目里写任何文件**（连测试报告都不写，见 -o addopts= 的说明）。
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from tools.logger import get_logger

logger = get_logger(__name__)

# pytest 退出码的含义（面试可能会问到，值得背下来）
EXIT_MEANING = {
    0: "全部通过",
    1: "有用例失败",
    2: "执行被中断",
    3: "内部错误",
    4: "命令行用法错误",
    5: "没有收集到任何用例",
}

# 从 pytest 输出里抓统计行，例如：
#   "3 passed, 1 failed in 12.34s"
#   "1 failed, 2 passed"
_STAT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")
# 抓失败的用例名，例如：
#   "FAILED tests/test_login.py::test_x - AssertionError: ..."
_FAILED_LINE = re.compile(r"^FAILED\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)


@dataclass
class TestRunResult:
    """一次 pytest 执行的结构化结果。"""

    command: list[str]
    returncode: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    summary_line: str = ""
    failed_tests: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    output_chars: int = 0
    timed_out: bool = False

    @property
    def exit_meaning(self) -> str:
        return EXIT_MEANING.get(self.returncode, f"未知退出码 {self.returncode}")

    @property
    def success(self) -> bool:
        """退出码 0 才算成功。"""
        return self.returncode == 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "退出码": f"{self.returncode}（{self.exit_meaning}）",
            "通过": self.passed,
            "失败": self.failed,
            "错误": self.errors,
            "跳过": self.skipped,
            "统计行": self.summary_line or "（未解析到）",
            "失败用例": self.failed_tests or "无",
            "输出长度": f"{self.output_chars} 字符",
        }

    def describe(self) -> str:
        if self.timed_out:
            return "执行超时"
        if not self.summary_line:
            return f"退出码 {self.returncode}（{self.exit_meaning}），未解析到统计行"
        return f"{self.summary_line} ｜ {self.exit_meaning}"


def run_pytest(
    target: str | Path | None = None,
    *,
    browser: str = "chromium",
    timeout: int = 300,
    extra_args: list[str] | None = None,
    python_exe: str | Path | None = None,
) -> TestRunResult:
    """执行 pytest，返回结构化结果。

    参数：
        target      要跑的文件或目录；默认跑 generated_tests/
        browser     只跑哪个浏览器。默认 chromium —— 对方项目默认跑
                    三种浏览器，一次执行要花三倍时间，演示时没必要
        timeout     超时秒数
        extra_args  额外的 pytest 参数（谨慎使用，会原样拼进命令）
        python_exe  用哪个解释器；默认用对方项目的 venv python
    """
    target_path = Path(target) if target else settings.GENERATED_DIR
    if not target_path.is_absolute():
        target_path = settings.PROJECT_ROOT / target_path

    interpreter = str(python_exe or settings.TEST_PROJECT_PYTHON)

    command = [
        interpreter,
        "-m", "pytest",
        str(target_path),
        # 关键：清空对方 pytest.ini 里预设的 addopts。
        # 对方的 addopts 里有 --html=reports/report.html 和 --alluredir=...，
        # 那会在**对方项目里写出测试报告文件**——我们约定不改动对方项目，
        # 所以这里显式清空，做到"只跑、不写"。
        "-o", "addopts=",
        # 只跑一种浏览器，加快演示速度
        "--browser", browser,
        "-q", "--no-header",
    ]
    if extra_args:
        command.extend(extra_args)

    # 工作目录切到对方项目根，并把它的根目录加进 PYTHONPATH。
    # 这样生成的代码里 `from pages.login_page import LoginPage` 才能 import 成功。
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{settings.TEST_PROJECT_DIR}{os.pathsep}{existing}" if existing
        else str(settings.TEST_PROJECT_DIR)
    )

    logger.info("执行 pytest（工作目录：%s）", settings.TEST_PROJECT_DIR)
    logger.info("命令：%s", " ".join(command))

    result = TestRunResult(command=command)

    try:
        completed = subprocess.run(
            command,
            cwd=str(settings.TEST_PROJECT_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result.timed_out = True
        result.returncode = -1
        result.summary_line = f"（执行超过 {timeout} 秒，已终止）"
        logger.error(result.summary_line)
        return result
    except OSError as exc:
        result.returncode = -1
        result.summary_line = f"无法启动 pytest：{type(exc).__name__}: {exc}"
        logger.error(result.summary_line)
        return result

    result.returncode = completed.returncode
    result.stdout = completed.stdout or ""
    result.stderr = completed.stderr or ""
    output = result.stdout + result.stderr
    result.output_chars = len(output)

    _parse_output(output, result)

    logger.info("pytest 结束：%s", result.describe())
    return result


def _parse_output(output: str, result: TestRunResult) -> None:
    """从 pytest 的原始输出里提取结构化信息。"""
    # 统计行：取最后一个包含 passed/failed/error 的行
    stat_lines = [
        line.strip() for line in output.splitlines()
        if _STAT_PATTERN.search(line)
    ]
    if stat_lines:
        result.summary_line = stat_lines[-1]

    for number, word in _STAT_PATTERN.findall(result.summary_line or output):
        value = int(number)
        if word == "passed":
            result.passed += value
        elif word == "failed":
            result.failed += value
        elif word in ("error", "errors"):
            result.errors += value
        elif word in ("skipped", "xfailed", "xpassed"):
            result.skipped += value

    # 失败用例名（截断过长的报错，避免刷屏）
    for name, reason in _FAILED_LINE.findall(output):
        reason = (reason or "").strip()
        if len(reason) > 120:
            reason = reason[:117] + "..."
        result.failed_tests.append(f"{name}  |  {reason}" if reason else name)
