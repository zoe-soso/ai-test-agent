"""
Day 21~27 测试

覆盖：
  Day 21  Allure 报告接入（generate_allure_report 优雅降级）
  Day 22  失败截图命名规则（与 failure_collector 对齐）
  Day 23  failure_collector 把 pytest 结果整理成"失败病历"
  Day 24/25 defect_analyzer 缺陷分析（分类 + 严重程度，含容错）
  Day 26  defect_agent 决策循环（重跑 → 分析 → 报告，离线可跑）
  Day 27  pipeline 全链路（pass / fail 两条分支，离线、不启动浏览器）

全部走 Mock 或桩函数，不联网、不启动浏览器。
"""

import json
from pathlib import Path

import pytest

from agent import defect_agent as defect_agent_mod
from agent.defect_analyzer import (
    DefectAnalysis, DefectAnalyzer, _norm_category, _norm_severity,
)
from agent.failure_collector import FailureRecord, collect_failures, _safe_name
from agent.llm_client import LLMClient
from agent.pipeline import run_pipeline
from config import settings
from tools.test_runner import TestRunResult


# ----------------------------------------------------------------------
# 共用：一个离线 Mock 客户端
# ----------------------------------------------------------------------
@pytest.fixture
def mock_client():
    return LLMClient(mock_mode="true")


# ----------------------------------------------------------------------
# Day 21：Allure
# ----------------------------------------------------------------------
def test_generate_allure_report_no_results_is_graceful():
    # 不存在的结果目录 -> 不应抛异常，返回 (False, 提示)
    ok, msg = __import__("tools.test_runner", fromlist=["generate_allure_report"]) \
        .generate_allure_report(results_dir=settings.OUTPUT_DIR / "_no_such_dir")
    assert ok is False
    assert "Allure" in msg


def test_allure_results_dir_exists():
    settings.ensure_dirs()
    assert settings.ALLURE_RESULTS_DIR.exists()
    assert settings.SCREENSHOT_DIR.exists()


# ----------------------------------------------------------------------
# Day 22：截图命名规则（必须和 conftest 里一致）
# ----------------------------------------------------------------------
def test_screenshot_name_is_filesystem_safe():
    assert _safe_name("tests/test_x.py::test_y") == "tests_test_x.py__test_y"
    assert _safe_name("a/b/c::test d") == "a_b_c__test_d"


# ----------------------------------------------------------------------
# Day 23：failure_collector
# ----------------------------------------------------------------------
def test_collect_failures_parses_one_failure_with_screenshot(tmp_path):
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir()
    # 放一张和 nodeid 对应命名的截图
    (shot_dir / "tests_test_x.py__test_login.png").write_text("fake")

    result = TestRunResult(command=["pytest"], passed=1, failed=1)
    result.summary_line = "1 failed, 1 passed in 2.00s"
    result.failed_tests = ["tests/test_x.py::test_login  |  AssertionError: 登录失败"]
    result.stdout = (
        "FAILED tests/test_x.py::test_login - AssertionError: 登录失败\n"
        "E   assert 'ok' in 'fail'\n"
        "1 failed, 1 passed in 2.00s"
    )

    records = collect_failures(result, screenshot_dir=shot_dir)
    assert len(records) == 1
    rec = records[0]
    assert rec.test_name == "tests/test_x.py::test_login"
    assert "登录失败" in rec.error
    assert "assert" in rec.traceback
    assert rec.screenshot is not None
    assert rec.screenshot.endswith("tests_test_x.py__test_login.png")


def test_collect_failures_no_failure_returns_empty():
    result = TestRunResult(command=["pytest"], passed=3, failed=0)
    result.summary_line = "3 passed"
    assert collect_failures(result) == []


# ----------------------------------------------------------------------
# Day 24/25：defect_analyzer
# ----------------------------------------------------------------------
def test_defect_analyzer_classifies_with_mock(mock_client):
    rec = FailureRecord(
        test_name="tests/test_x.py::test_login",
        error="AssertionError",
        traceback="E AssertionError: assert 'ok' in 'fail'",
    )
    analysis = DefectAnalyzer(mock_client).analyze(rec)
    # mock 预设：元素定位问题 / P1
    assert analysis.category == "元素定位问题"
    assert analysis.severity == "P1"
    assert analysis.possible_causes
    assert analysis.suggestions


def test_norm_category_and_severity_aliases():
    assert _norm_category("UI") == "UI问题"
    assert _norm_category("接口问题") == "接口问题"
    assert _norm_category("乱写的") == "未知"
    assert _norm_severity("p0") == "P0"
    assert _norm_severity("高") == "P1"
    assert _norm_severity("zzz") == "P1"  # 未知归一为默认 P1


def test_defect_analyzer_falls_back_on_bad_json(mock_client, monkeypatch):
    # 让 chat_messages 返回一个非法 JSON，验证不崩、降级
    def bad(_messages, mock_hint=None):
        return "这不是 JSON"
    monkeypatch.setattr(mock_client, "chat_messages", bad)
    rec = FailureRecord(test_name="t", error="e", traceback="tb")
    analysis = DefectAnalyzer(mock_client).analyze(rec)
    assert analysis.category == "未知"
    assert analysis.severity == "P1"


# ----------------------------------------------------------------------
# Day 26：defect_agent 决策循环
# ----------------------------------------------------------------------
def _make_failing_result() -> TestRunResult:
    r = TestRunResult(command=["pytest"], passed=42, failed=1)
    r.summary_line = "1 failed, 42 passed in 3.21s"
    r.failed_tests = ["tests/test_x.py::test_login  |  AssertionError: 登录后未跳转"]
    r.stdout = "FAILED tests/test_x.py::test_login - AssertionError: 登录后未跳转\n1 failed, 42 passed in 3.21s"
    return r


def test_defect_agent_loop_runs_offline(mock_client):
    # 用桩 runner，避免真的启动浏览器
    def stub_runner(target, **kw):
        r = TestRunResult(command=["pytest"], passed=42, failed=1)
        r.summary_line = "1 failed, 42 passed in 3.21s"
        return r

    agent = defect_agent_mod.DefectAnalysisAgent(mock_client, runner=stub_runner)
    report = agent.analyze_run(_make_failing_result(), feature="登录", max_iterations=5)
    assert report.failed == 1
    assert len(report.analyses) == 1
    assert report.analyses[0].category in (
        defect_agent_mod.DefectAnalysisAgent.__module__ and ["元素定位问题"]
    )
    assert "重跑" in report.rerun_info  # 证明模型先"决策重跑"


def test_defect_agent_no_failure_short_circuits(mock_client):
    ok = TestRunResult(command=["pytest"], passed=5, failed=0)
    ok.summary_line = "5 passed"
    agent = defect_agent_mod.DefectAnalysisAgent(mock_client, runner=lambda *a, **k: ok)
    report = agent.analyze_run(ok, feature="登录")
    assert report.analyses == []
    assert "全部通过" in report.summary


# ----------------------------------------------------------------------
# Day 27：pipeline 全链路（离线，不启动浏览器）
# ----------------------------------------------------------------------
def test_run_pytest_resolves_relative_dotdot_target(monkeypatch):
    """回归：AI 重跑工具给来的相对路径(如 '..\\...\\generated_tests\\x.py')
    必须解析成干净的绝对路径，不能和 PROJECT_ROOT 重复拼接成
    '甲\\..\\甲\\...'（那会让 pytest 报"命令行用法错误 exit 4"）。"""
    import subprocess
    from tools import test_runner

    captured = {}
    rel = r"..\基于 LLM 的 Web 智能测试用例生成与缺陷分析 Agent\ai-test-agent\generated_tests\test_tc_login_001.py"

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    test_runner.run_pytest(rel, timeout=5)

    target = captured["cmd"][3]  # [python, -m, pytest, <target>, ...]
    # 必须以本项目 generated_tests/ 开头
    assert target.startswith(str(settings.GENERATED_DIR))
    # 归一化后不允许再出现 ..\ 片段
    assert "\\..\\" not in target
    # 且该文件真实存在（相对对方项目根解析后能落到我们自己的文件）
    assert Path(target).exists()


def test_pipeline_pass_branch_offline(mock_client, monkeypatch, tmp_path):
    # 桩掉代码生成与执行环节，让流程确定可控、不启动浏览器
    # 并把"生成代码落盘目录"指到临时目录，避免测试真的写进 generated_tests/
    from agent.code_generator import GeneratedCode
    monkeypatch.setattr("config.settings.GENERATED_DIR", tmp_path / "gen")
    fake_code = GeneratedCode(
        feature="用户登录功能", case_id="TC_001", case_name="正常登录",
        code="def test_tc_001(page_context):\n    page_context.goto('/')\n    assert 'Login' in page_context.content()\n",
        filename="test_tc_001.py", issues=[],
    )
    monkeypatch.setattr(
        "agent.pipeline.CodeGenerator.generate_many",
        lambda *a, **k: [fake_code],
    )

    def fake_run_pytest(target, **kw):
        r = TestRunResult(command=["pytest"], passed=1, failed=0)
        r.summary_line = "1 passed"
        return r
    monkeypatch.setattr("tools.test_runner.run_pytest", fake_run_pytest)

    report = run_pipeline(
        mock_client, feature="用户登录功能", limit=1, auto=True,
    )
    assert report.cases_count >= 1
    assert len(report.code_files) >= 1
    assert report.exec_result.get("通过") == 1
    assert report.defect_report_path == ""  # 全通过，不产缺陷报告


def test_pipeline_fail_branch_runs_analysis_offline(mock_client, monkeypatch, tmp_path):
    from agent.code_generator import GeneratedCode
    monkeypatch.setattr("config.settings.GENERATED_DIR", tmp_path / "gen")
    fake_code = GeneratedCode(
        feature="用户登录功能", case_id="TC_001", case_name="正常登录",
        code="def test_tc_001(page_context):\n    page_context.goto('/')\n    assert False\n",
        filename="test_tc_001.py", issues=[],
    )
    monkeypatch.setattr(
        "agent.pipeline.CodeGenerator.generate_many",
        lambda *a, **k: [fake_code],
    )

    def fake_run_pytest(target, **kw):
        r = _make_failing_result()
        r.returncode = 1  # 关键：非零退出码才代表"有失败"
        return r
    # 缺陷分析 Agent 也会"重跑"，同样要桩掉（runner 会一路传给
    # DefectAnalysisAgent），否则它会真的启动浏览器去跑（曾因此挂死测试）。
    monkeypatch.setattr("tools.test_runner.run_pytest", fake_run_pytest)

    report = run_pipeline(
        mock_client, feature="用户登录功能", limit=1, auto=True,
        runner=fake_run_pytest,
    )
    assert report.exec_result.get("失败") == 1
    assert report.defect_report_path  # 失败分支产出了缺陷分析报告
    # 报告文件真的写出来了
    assert Path(report.defect_report_path).exists()
