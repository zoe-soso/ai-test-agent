"""
Day 28 单元测试：目标项目档案（ProjectProfile）

覆盖：
  - 档案发现（available_profiles）
  - 加载（load_profile：命中 / 找不到回落）
  - 功能 -> 页面对象 解析（resolve_page：关键词命中 / 兜底）
  - 静态方法清单读取（load_methods，需被测项目在才跑）
  - 自动发现（discover_pages：目录不存在时优雅返回空）
  - 模块内部工具（_class_methods：纯离线解析）

全部不联网；需要被测项目的测试做了 skip 保护（CI 上没装也能绿）。
"""

import pytest

from agent import project_profile
from config import settings


# ----------------------------------------------------------------------
# 档案发现 / 加载
# ----------------------------------------------------------------------
def test_available_profiles_includes_ecommerce():
    names = project_profile.available_profiles()
    # ecommerce.yaml 是内置的电商项目档案
    assert "ecommerce" in names


def test_load_profile_ecommerce_returns_pages():
    profile = project_profile.load_profile("ecommerce")
    # ecommerce.yaml 里配了 pages 映射
    assert profile.pages, "ecommerce 档案应至少含一个页面映射"
    assert profile.display_name
    # pages_root 是绝对路径（YAML 里写的是绝对路径）
    assert profile.pages_root.is_absolute()


def test_load_profile_missing_falls_back_to_default():
    # 找不到的档案名 -> 回落内置默认（不报错、不崩）
    profile = project_profile.load_profile("这个档案一定不存在_xyz")
    assert profile.name == "default"
    assert profile.project_dir == settings.TEST_PROJECT_DIR


def test_load_profile_env_var_drives_name(monkeypatch):
    # 不传名字时，应读 AI_AGENT_PROFILE 环境变量
    monkeypatch.setenv("AI_AGENT_PROFILE", "ecommerce")
    profile = project_profile.load_profile()
    assert profile.pages


# ----------------------------------------------------------------------
# 功能 -> 页面对象 解析
# ----------------------------------------------------------------------
def test_resolve_page_login_matches_keyword():
    profile = project_profile.load_profile("ecommerce")
    spec = profile.resolve_page("用户登录功能")
    assert spec.class_name == "LoginPage"
    assert spec.module == "pages.login_page"


def test_resolve_page_unknown_uses_fallback():
    profile = project_profile.load_profile("ecommerce")
    spec = profile.resolve_page("一个完全陌生、档案里没有的功能")
    # ecommerce.yaml 声明了 fallback_page = LoginPage
    assert spec.class_name == "LoginPage"


# ----------------------------------------------------------------------
# 静态方法清单（需要被测项目在磁盘上）
# ----------------------------------------------------------------------
def test_load_methods_reads_real_source():
    profile = project_profile.load_profile("ecommerce")
    if not profile.exists:
        pytest.skip("找不到关联的被测项目，跳过")
    methods = profile.load_methods("pages.login_page")
    assert "login" in methods
    assert "open" in methods            # 来自 BasePage 继承
    assert "fill_email" not in methods  # 编造的方法不该出现


# ----------------------------------------------------------------------
# 自动发现：不配 pages 时扫描目标项目 pages/ 目录
# ----------------------------------------------------------------------
def test_discover_pages_missing_dir_returns_empty(tmp_path):
    # 一个指向不存在目录的档案，discover 应优雅返回 []（不抛异常）
    profile = project_profile.ProjectProfile(name="x", project_dir=tmp_path / "nope")
    assert profile.discover_pages() == []


def test_discover_pages_ignores_base_classes(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    # 基类不该被当成页面对象
    (pages / "base_page.py").write_text(
        "class BasePage:\n    def open(self): ...\n"
    )
    # 真正的页面对象
    (pages / "login_page.py").write_text(
        "class LoginPage:\n    def login(self): ...\n    def get_user(self): ...\n"
    )
    profile = project_profile.ProjectProfile(name="x", project_dir=tmp_path)
    specs = profile.discover_pages()
    classes = {s.class_name for s in specs}
    assert "LoginPage" in classes
    assert "BasePage" not in classes       # 基类被忽略


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def test_class_methods_parses_temp_file(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(
        "class CartPage:\n"
        "    def open(self): ...\n"
        "    def add(self): ...\n"
        "    def _hidden(self): ...\n"
    )
    result = project_profile._class_methods(f)
    assert "CartPage" in result
    methods = result["CartPage"]
    assert "open" in methods
    assert "add" in methods


def test_format_methods_truncates_long_list():
    methods = {f"m{i}" for i in range(20)}
    text = project_profile._format_methods(methods, limit=3)
    assert "等" in text                       # 超过 limit 会提示"等 N 个方法"
    assert "m0" in text
