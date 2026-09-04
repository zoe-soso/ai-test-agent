"""
缺陷分析 Agent（Day 26）

这一天是"真正体现 Agent"的一天。

前面 Day 24/25 的 DefectAnalyzer 只是"给一条失败，返回一个分析"，
是一次性函数调用。Day 26 要把它升级成**会自己决策**的 Agent：

    测试失败
        ↓
    Agent 观察失败信息
        ↓
    Agent 决策：要不要先重跑一次排除偶发？
        ↓ 要
    调用 rerun_test 工具
        ↓
    Agent 决策：现在分析失败原因
        ↓
    调用 analyze_failure 工具
        ↓
    调用 finalize_report 工具
        ↓
    产出"智能测试报告"

"观察 → 决策 → 调用工具"这个循环，就是 Agent 和普通"调一次 LLM"的本质区别。
实现上复用 Day 13 的 run_tool_loop —— 只要把工具注册进去，循环逻辑是一样的。

为什么重跑一次？
    自动化测试有"偶发失败"（flaky）：网络抖一下、浏览器慢半拍，
    同一段代码点两次有一次就过。这种失败重跑一次往往就绿了，
    不该提缺陷单。让 Agent 先重跑、再判断，比"一失败就报警"专业得多。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import tool_calling
from agent.defect_analyzer import DefectAnalysis, DefectAnalyzer
from agent.failure_collector import (
    FailureRecord, collect_failures, summarize,
)
from config import settings
from tools import file_io
from tools.logger import get_logger
from tools.test_runner import TestRunResult, run_pytest

logger = get_logger(__name__)


@dataclass
class DefectReport:
    """一次测试执行后的"智能测试报告"。"""

    feature: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    analyses: list[DefectAnalysis] = field(default_factory=list)
    summary: str = ""
    rerun_info: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "analyses": [a.to_dict() for a in self.analyses],
            "summary": self.summary,
            "rerun_info": self.rerun_info,
        }

    def save(self, path: str | None = None) -> str:
        """把报告存成 YAML（Day 27 流水线会调用）。"""
        out = Path(path or (settings.OUTPUT_DIR / "defect_report.yaml"))
        file_io.write_yaml(out, self.to_dict())
        return str(out)


def _failure_from_json(text: str) -> FailureRecord:
    """把工具参数里的 failure_json 还原成 FailureRecord（容错）。"""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return FailureRecord(test_name="未知用例", error=str(text)[:200])
    return FailureRecord(
        test_name=str(data.get("test_name", "未知用例")),
        error=str(data.get("error", "")),
        traceback=str(data.get("traceback", data.get("error", ""))),
        screenshot=str(data.get("screenshot")) if data.get("screenshot") else None,
    )


class DefectAnalysisAgent:
    """观察失败 → 决策重跑/分析 → 产出报告的 Agent。"""

    def __init__(
        self,
        client: Any,
        *,
        analyzer: DefectAnalyzer | None = None,
        runner: Any | None = None,
        browser: str = "chromium",
        timeout: int = 300,
    ) -> None:
        """
        client   带 chat_with_tools 的 LLM 客户端（真实或 Mock）
        analyzer 缺陷分析器；不传则用 client 现建一个
        runner   重跑测试的函数；默认 tools.test_runner.run_pytest
                 测试时可传入桩函数，避免真的启动浏览器
        """
        self.client = client
        self.analyzer = analyzer or DefectAnalyzer(client)
        self.runner = runner or run_pytest
        self.browser = browser
        self.timeout = timeout
        self._registry = self._build_registry()

    def _build_registry(self) -> tool_calling.ToolRegistry:
        reg = tool_calling.ToolRegistry()

        @reg.tool
        def rerun_test(target: str, timeout: int = 300) -> str:
            """重新执行某个测试文件/目录，用于排除偶发性失败。

            Args:
                target: 要重跑的文件或目录路径
                timeout: 超时秒数
            """
            res = self.runner(target, browser=self.browser, timeout=timeout)
            return f"重跑完成：{res.describe()}"

        @reg.tool
        def analyze_failure(failure_json: str) -> str:
            """对一条失败用例做 AI 缺陷分析，返回分类、严重程度与原因。

            Args:
                failure_json: 失败病历的 JSON 字符串（含 test_name/error/traceback）
            """
            rec = _failure_from_json(failure_json)
            analysis = self.analyzer.analyze(rec)
            return json.dumps(analysis.to_dict(), ensure_ascii=False)

        @reg.tool
        def finalize_report(summary: str) -> str:
            """记录最终的缺陷分析结论，结束流程。

            Args:
                summary: 最终结论文字
            """
            self._final_summary = summary
            return f"报告已记录：{summary[:200]}"

        return reg

    def analyze_run(
        self,
        result: TestRunResult,
        *,
        feature: str = "",
        max_iterations: int = 5,
    ) -> DefectReport:
        """对一次 pytest 结果做缺陷分析，返回结构化报告。

        如果没有任何失败，直接返回"全部通过"，不调 LLM、不花一分钱。
        """
        report = DefectReport(
            feature=feature,
            total=result.total,
            passed=result.passed,
            failed=result.failed,
        )

        records = collect_failures(result)
        if not records:
            report.summary = "全部通过，无缺陷。"
            return report

        # 把失败病历压成一段文字，作为 Agent 的"观察"
        user_text = (
            f"测试执行完成，有 {len(records)} 条失败。\n"
            "请先判断是否要重跑排除偶发，再分析失败原因，最后给出结论。\n\n"
            f"{summarize(records)}"
        )
        messages = [{"role": "user", "content": user_text}]

        loop = tool_calling.run_tool_loop(
            self.client, messages, self._registry, max_iterations=max_iterations,
        )

        # 从循环步骤里收集 analyze_failure 的产物，组装成 DefectAnalysis
        for step in loop.steps:
            if step.tool == "analyze_failure" and step.result and not step.error:
                try:
                    data = json.loads(step.result)
                    report.analyses.append(DefectAnalysis(
                        test_name=str(data.get("test_name", "")),
                        problem_summary=str(data.get("problem_summary", "")),
                        category=str(data.get("category", "未知")),
                        severity=str(data.get("severity", "P1")),
                        possible_causes=list(data.get("possible_causes", [])),
                        suggestions=list(data.get("suggestions", [])),
                    ))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("analyze_failure 返回无法解析：%s", step.result[:200])

        # 有没有走过"重跑"这步，记到报告里
        if "rerun_test" in loop.tool_names:
            report.rerun_info = "Agent 先重跑一次以排除偶发失败。"

        report.summary = getattr(self, "_final_summary", loop.answer)
        return report
