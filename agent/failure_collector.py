"""
失败信息收集器（Day 23）

为什么需要它？
    从 Day 20 起，Agent 能"跑 pytest 并拿到结果"。但 `TestRunResult`
    只告诉我们"哪条失败了、错在哪一行"，信息太散：
        - 失败用例名在 failed_tests
        - 真实报错（AssertionError 详情、堆栈）藏在几 KB 的原始输出里
        - 截图在另一个目录

    做失败分析（Day 24~26）之前，必须先把这些信息**统一成一份结构化的
    "病历"**。这就是 failure_collector 的工作：

        pytest 原始结果 + 截图目录
                ↓
        [FailureRecord, FailureRecord, ...]

    每个 FailureRecord 就是一条"失败病历"，包含：
        用例名 / 错误信息 / 堆栈 / 截图路径 / 相关日志

设计要点：
    1. 纯函数、无副作用、可单测。它只读文件、解析文本，不调 LLM、不联网。
    2. 找不到截图/日志也不报错，只是那一项为 None —— 优雅降级。
    3. 解析原始输出用"按 nodeid 切分"，不依赖任何 pytest 插件，
       所以即使对方项目没装 json-report，也能工作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from tools.test_runner import TestRunResult

# 把 nodeid（如 tests/test_x.py::test_y）变成文件名安全字符串，
# 必须和 generated_tests/conftest.py 里的 _safe_name 规则一致，
# 否则"失败信息"和"截图"对不上号。
_SAFE = re.compile(r"[^\w.\-]")


def _safe_name(nodeid: str) -> str:
    return _SAFE.sub("_", nodeid.replace("::", "__").replace("/", "_"))


@dataclass
class FailureRecord:
    """一条"失败病历"。"""

    test_name: str                       # 完整 nodeid，如 generated_tests/test_x.py::test_y
    error: str = ""                      # 一句话的错误（来自 FAILED 行）
    traceback: str = ""                  # 完整报错 + 堆栈（从 pytest 输出里切出来的）
    screenshot: str | None = None        # 截图文件路径（没有就是 None）
    log: str = ""                        # 相关日志片段（目前取 traceback，后续可扩展）

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "error": self.error,
            "traceback": self.traceback,
            "screenshot": self.screenshot,
            "log": self.log or self.traceback,
        }


def _split_nodeid_and_reason(entry: str) -> tuple[str, str]:
    """failed_tests 里一项长这样：'nodeid  |  原因'。拆开。"""
    if "  |  " in entry:
        name, _, reason = entry.partition("  |  ")
        return name.strip(), reason.strip()
    return entry.strip(), ""


def _canonical_nodeid(display_nodeid: str) -> str:
    r"""把 pytest 打印的"显示型 nodeid"（可能带 ..\ 前缀和空格）还原成
    conftest 里 item.nodeid 用的规范形式（generated_tests/...，正斜杠）。

    截图文件名正是用规范 nodeid 算出来的，所以这里必须对得上。
    """
    idx = display_nodeid.find("generated_tests")
    sub = display_nodeid[idx:] if idx >= 0 else display_nodeid
    return sub.replace("\\", "/")


def _test_basename(display_nodeid: str) -> str:
    """取用例文件名（不含空格），用于在原始输出里定位报错块。

    例如 '..\\基于 LLM 的 ...\\generated_tests\\test_x.py::test_y[chromium]'
    -> 'test_x.py'。用文件名搜而不用完整 nodeid，避开路径里的空格。
    """
    core = re.split(r"[\\/]", display_nodeid)[-1]
    core = core.split("::", 1)[0].split("[", 1)[0]
    return core


def _extract_traceback(output: str, nodeid: str, max_lines: int = 40) -> str:
    """从 pytest 原始输出里，切出属于这条用例的报错块。

    思路：用文件名（不含空格）定位到这条用例的报错区，往下抓，
    直到遇到下一条 FAILED 或统计行（如 '1 failed'）或文件结尾为止。
    """
    basename = _test_basename(nodeid)
    lines = output.splitlines()
    start = None
    for index, line in enumerate(lines):
        if basename in line:
            start = index
            break
    if start is None:
        return ""

    block: list[str] = []
    for line in lines[start:start + max_lines]:
        # 遇到下一条失败的用例名，停止（避免把别人的报错也带进来）
        if line.startswith("FAILED ") and basename not in line:
            break
        # 遇到统计汇总行（"N passed, M failed in Xs"），停止
        if re.match(r"^\s*\d+\s+(passed|failed|error)", line):
            break
        block.append(line)
    return "\n".join(block).strip()


def collect_failures(
    result: TestRunResult,
    screenshot_dir: str | Path | None = None,
) -> list[FailureRecord]:
    """把一次 pytest 结果的失败信息，整理成结构化的 FailureRecord 列表。

    参数：
        result          来自 tools.test_runner.run_pytest 的结果
        screenshot_dir  截图目录；默认用 settings.SCREENSHOT_DIR

    返回：
        FailureRecord 列表。一条都没失败就返回空列表。
    """
    if result.failed == 0 and not result.failed_tests:
        return []

    shot_dir = Path(screenshot_dir or settings.SCREENSHOT_DIR)
    output = result.stdout + "\n" + result.stderr

    records: list[FailureRecord] = []
    for entry in result.failed_tests:
        nodeid, reason = _split_nodeid_and_reason(entry)
        traceback = _extract_traceback(output, nodeid)

        # 截图文件名要和 conftest 里的命名规则一致：
        # conftest 用 item.nodeid（规范形式 generated_tests/...）算名字，
        # 所以这里把 pytest 显示的 nodeid 也还原成规范形式再算。
        shot_path: str | None = None
        candidate = shot_dir / f"{_safe_name(_canonical_nodeid(nodeid))}.png"
        if candidate.exists():
            shot_path = str(candidate)

        records.append(FailureRecord(
            test_name=nodeid,
            error=reason,
            traceback=traceback or reason,
            screenshot=shot_path,
            log=traceback or reason,
        ))
    return records


def summarize(records: list[FailureRecord]) -> str:
    """把所有失败病历压成一段给 LLM 看的文字（Day 24 用）。"""
    if not records:
        return "（没有失败）"
    parts: list[str] = []
    for index, rec in enumerate(records, 1):
        head = f"【失败 {index}】{rec.test_name}"
        body = rec.traceback or rec.error
        # 截断超长 traceback，避免把 LLM 的上下文撑爆
        if len(body) > 1500:
            body = body[:1500] + "\n...（已截断）"
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)
