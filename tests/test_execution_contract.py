from types import SimpleNamespace
import subprocess

import pytest

import main
from agent import llm_client, pipeline, project_profile
from agent.code_generator import CodeGenerator, GeneratedCode, validate_code
from config import settings
from tools import test_runner


@pytest.fixture
def profile(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "base_page.py").write_text("class BasePage:\n    def open(self, url): pass\n")
    (pages / "login_page.py").write_text(
        "from .base_page import BasePage\n"
        "class LoginPage(BasePage):\n"
        "    def login(self, email, password): pass\n"
        "    def is_login_success(self, username): pass\n"
        "    def optional(self, value=None, *, enabled=True): pass\n"
        "class Helper:\n    def invented(self): pass\n"
    )
    return project_profile.ProjectProfile(project_dir=tmp_path)


@pytest.mark.parametrize("call,invalid", [
    ("is_login_success()", True),
    ("is_login_success('zoe')", False),
    ("login('email', password='pwd')", False),
    ("login('email', 'pwd', 'extra')", True),
    ("login('email', typo='pwd')", True),
    ("open()", True),
    ("optional(enabled=False)", False),
    ("invented()", True),
])
def test_page_contract_uses_real_class_and_signature(profile, call, invalid):
    code = ("from pages.login_page import LoginPage\n"
            "def test_login(page_context):\n"
            "    login_page = LoginPage(page_context)\n"
            "    login_page.open('https://example.com')\n"
            f"    login_page.{call}\n")
    issues = validate_code(code, profile=profile)
    assert bool(issues) is invalid


def prepare_pipeline(monkeypatch, tmp_path, generated):
    monkeypatch.setattr(settings, "GENERATED_DIR", tmp_path / "generated")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "reports")
    monkeypatch.setattr(pipeline._planner, "TestCaseAgent", lambda **kw: SimpleNamespace(
        run=lambda *a: SimpleNamespace(suite={"cases": [{}]}, revise_rounds=0)))
    monkeypatch.setattr(CodeGenerator, "generate_many", lambda *a, **kw: generated)


def test_invalid_generation_never_saved_or_executed(monkeypatch, tmp_path):
    invalid = GeneratedCode("login", "bad", "bad", code="invalid !!!", filename="test_bad.py", issues=["syntax"])
    prepare_pipeline(monkeypatch, tmp_path, [invalid])
    report = pipeline.run_pipeline(object(), feature="login", auto=True,
                                   runner=lambda *a, **kw: pytest.fail("invalid code executed"))
    assert report.status == "generation_failed"
    assert report.exit_code != 0
    assert not (settings.GENERATED_DIR / "test_bad.py").exists()
    assert (settings.OUTPUT_DIR / "pipeline_report.yaml").exists()
    with pytest.raises(ValueError):
        CodeGenerator(client=object()).save(invalid)


def test_multi_file_run_excludes_stale_files(monkeypatch, tmp_path):
    generated = [GeneratedCode("login", str(i), "test", code="def test_ok(): pass", filename=f"test_new_{i}.py")
                 for i in range(2)]
    prepare_pipeline(monkeypatch, tmp_path, generated)
    settings.GENERATED_DIR.mkdir()
    stale = settings.GENERATED_DIR / "test_stale.py"
    stale.write_text("raise RuntimeError('must not run')")
    captured = []
    def runner(target, **kw):
        captured.extend(target)
        return test_runner.TestRunResult(command=[], passed=2)
    report = pipeline.run_pipeline(object(), feature="login", limit=2, auto=True, runner=runner)
    assert set(captured) == set(report.code_files)
    assert str(stale) not in captured
    assert report.exit_code == 0
    assert (settings.OUTPUT_DIR / "pipeline_report.yaml").exists()


@pytest.mark.parametrize("status,expected", [("passed", 0), ("execution_failed", 1), ("cancelled", 2), ("generation_failed", 3)])
def test_cli_propagates_pipeline_status(monkeypatch, status, expected):
    monkeypatch.setattr(llm_client, "get_client", lambda: SimpleNamespace(is_mock=True))
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **kw: pipeline.PipelineReport(status=status))
    args = SimpleNamespace(profile="ecommerce", feature="login", description="", limit=1,
                           auto=True, browser="chromium", timeout=10, allure=False, max_repairs=0)
    assert main.cmd_run(args) == expected


def test_runner_passes_each_selected_file_to_pytest(monkeypatch, tmp_path):
    captured = {}
    def fake_run(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        return subprocess.CompletedProcess(command, 0, stdout="2 passed in 1s", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    selected = [tmp_path / "test_a.py", tmp_path / "test_b.py"]
    result = test_runner.run_pytest(selected)
    assert captured["command"][3:5] == [str(path.resolve()) for path in selected]
    assert captured["env"]["TEST_ARTIFACT_DIR"].startswith(str(settings.OUTPUT_DIR))
    assert result.passed == 2


def test_profile_environment_overrides_machine_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_PROJECT_DIR", str(tmp_path))
    profile = project_profile.load_profile("ecommerce")
    assert profile.project_dir == tmp_path
