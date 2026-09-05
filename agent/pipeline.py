"""
完整流水线（Day 27）—— 整个项目的核心 Demo

把前面所有模块串成一条线：

        用户需求
           ↓
      AI 需求分析        （Day 9，TestCaseGenerator）
           ↓
      测试用例生成
           ↓
       用例 Review          （Day 11，reviewer）
           ↓
      Playwright 代码       （Day 17，code_generator）
           ↓
       人工确认             （Day 19，human）
           ↓
       pytest 执行          （Day 20，test_runner）
           ↓
   ┌──────────┴──────────┐
   ↓ PASS                ↓ FAIL
 测试报告         截图 + 日志收集   （Day 22/23）
                      ↓
                AI 失败分析          （Day 24/25，defect_analyzer）
                      ↓
                智能测试报告          （Day 26，defect_agent）

这就是"从测试需求到执行和缺陷分析的闭环"。
面试时现场跑这条命令，就是你的 Demo。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import planner as _planner
from agent.code_generator import CodeGenerator
from agent.defect_agent import DefectAnalysisAgent, DefectReport
from agent import project_profile
from config import settings
from tools import file_io, human, test_runner
from tools.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineReport:
    """一次完整流水线的结果汇总。"""

    feature: str = ""
    cases_count: int = 0
    code_files: list[str] = field(default_factory=list)
    exec_result: dict[str, Any] = field(default_factory=dict)
    defect_report_path: str = ""
    full: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | None = None) -> str:
        out = Path(path or (settings.OUTPUT_DIR / "pipeline_report.yaml"))
        file_io.write_yaml(out, self.full)
        return str(out)


def run_pipeline(
    client: Any,
    *,
    feature: str,
    description: str = "（无额外描述）",
    limit: int = 1,
    auto: bool = False,
    browser: str = "chromium",
    timeout: int = 300,
    allure: bool = False,
    max_repairs: int = 1,
    runner: Any | None = None,
    profile: project_profile.ProjectProfile | None = None,
) -> PipelineReport:
    """跑完整条流水线，返回汇总。

    参数：
        client    LLM 客户端（真实或 Mock）
        feature   功能名，如"用户登录功能"
        limit     最多生成几条用例对应的代码（控制演示耗时/费用）
        auto      True 时跳过"人工确认"直接执行（演示/CI 用；
                  默认 False，遵守 Day 19 的 Human-in-the-loop）
        runner    执行 pytest 的函数；默认 tools.test_runner.run_pytest。
                  传给缺陷分析 Agent 用（测试可传桩函数，避免真开浏览器）。
        profile   目标项目档案（Day 28）。传给 CodeGenerator 和 run_pytest，
                  让"生成代码"和"执行"都按同一份档案走，接新项目不必改代码。
    """
    report = PipelineReport(feature=feature)

    # ---- 1~3. 需求 → 用例 → Review → 落盘（复用 Day 14 的 Agent）----
    print("【1/6】生成测试用例并评审 ...")
    agent = _planner.TestCaseAgent(client=client, max_repairs=max_repairs)
    agent_result = agent.run(feature, description)
    cases = agent_result.suite["cases"]
    report.cases_count = len(cases)
    print(f"       生成 {len(cases)} 条，评审轮数 {agent_result.revise_rounds}")

    # ---- 4. 用例 → Playwright 代码 ----
    print("【2/6】生成 Playwright 代码 ...")
    generator = CodeGenerator(client=client, max_repairs=max_repairs, profile=profile)
    generated = generator.generate_many(feature, cases, limit=limit)
    print(f"       代码生成 {len(generated)} 个")

    # ---- 5. 检查 + 保存 ----
    saved: list[str] = []
    for item in generated:
        for issue in item.issues:
            logger.info("代码问题[%s]：%s", item.filename, issue)
        if not item.code:
            continue
        if auto or human.ask_yes_no(f"是否保存 {item.filename}？", default=True):
            path = generator.save(item)
            saved.append(str(path))
    report.code_files = saved
    print(f"【3/6】保存 {len(saved)} 个代码文件")

    if not saved:
        print("       没有可执行的代码，流水线在生成阶段结束。")
        report.full = report.__dict__.copy()
        return report

    # ---- 6. 人工确认（Day 19）----
    # 只跑"本次新保存的文件"，不跑整个 generated_tests/ 目录，
    # 否则上次残留的旧测试文件也会被一起执行、污染结果。
    dir_for_confirm = str(Path(saved[0]).parent)
    if not auto:
        code_text = "\n\n".join(Path(p).read_text(encoding="utf-8") for p in saved)
        if not human.confirm_execution(code_text, dir_for_confirm):
            print("       你选择不执行。代码已保存，可随时手动跑。")
            report.full = report.__dict__.copy()
            return report

    # ---- 7. pytest 执行（Day 20）----
    # 只跑一条(默认 limit=1)就传单个文件；多条时跑本次文件的目录。
    # 这里 saved 里的路径都是绝对路径(CodeGenerator.save 返回)，可直接交给 run_pytest。
    print(f"【4/6】执行 pytest（浏览器 {browser}）...")
    exec_target = saved[0] if len(saved) == 1 else dir_for_confirm
    result = test_runner.run_pytest(
        exec_target, browser=browser, timeout=timeout, allure=allure, profile=profile,
    )
    report.exec_result = result.to_dict()
    print(f"       结果：{result.describe()}")

    # ---- 8. 失败分析（Day 22~26）----
    if result.success:
        print("【5/6】全部通过，无需缺陷分析。")
        report.full = report.__dict__.copy()
        return report

    print("【5/6】收集失败信息 + AI 缺陷分析 ...")
    defect_agent = DefectAnalysisAgent(client, runner=runner)
    defect_report: DefectReport = defect_agent.analyze_run(
        result, feature=feature, max_iterations=5,
    )
    report.defect_report_path = defect_report.save()
    print(f"       分析报告已保存：{report.defect_report_path}")
    for a in defect_report.analyses:
        print(f"       - {a.test_name}: {a.category}({a.severity})")

    # ---- 收尾 ----
    report.full = {
        "feature": report.feature,
        "cases_count": report.cases_count,
        "code_files": report.code_files,
        "exec_result": report.exec_result,
        "defect_report": defect_report.to_dict(),
    }
    out = report.save()
    print(f"\n【6/6】流水线完成，汇总报告：{out}")
    return report
