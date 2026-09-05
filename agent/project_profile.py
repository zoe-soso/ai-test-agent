"""
目标项目档案（Day 28：工程化 / 解耦）

## 为什么要有这个模块？

在此之前，"要接哪个被测项目"这件事被**散落在 4 个地方**硬编码：

    1. config/settings.py      TEST_PROJECT_DIR（目标项目路径）
    2. agent/code_generator.py PAGE_REGISTRY（"登录"→LoginPage 的中文映射表）
    3. agent/code_generator.py load_page_methods 里的 pages/ 与 base_page.py
    4. generated_tests/conftest.py 又写了一遍对方项目路径（两处真值不同步！）

结果就是：想接一个新项目，得改 4 个地方、还容易漏。更要命的是第 4 点——
**同一个路径写了两份**，改一处忘一处就会出现"用例生成按 A 项目、执行按 B 项目"的
诡异问题。

## 这个模块做什么

把上述 4 处收拢成**一份 YAML 档案** `config/profiles/<name>.yaml`：

    config/profiles/
    ├── ecommerce.yaml   ← 当前电商项目
    └── your_project.yaml ← 换项目：照抄一份改改即可，代码一行不动

## 关键设计：pages 可以"不写"，自动发现

档案里的 `pages:` 是**可选**的。不写时，本模块会去目标项目的 `pages/` 目录
用 ast 静态扫描所有 `*Page` 类，把真实方法清单全部读出来。

这意味着：**接一个全新的 POM 项目，你只需要填"项目在哪、解释器在哪"两行**，
页面类和方法清单是自动发现的（也自动防了模型编方法名的幻觉）。

## 仍然建议手写 pages 映射的原因

自动发现能拿到"有哪些类、每个类有哪些方法"，但拿不到
"中文需求里的'登录'两个字对应哪个类"。所以：

- 写了 `pages:` 映射 → 走**关键词精确匹配**（快、准、省 token）
- 没写 → 走**自动发现 + 交给模型按功能挑**（通用，但多花一点 token）

两者都支持，按需选择。这就是"约定优于配置"。
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from tools import file_io
from tools.logger import get_logger

logger = get_logger(__name__)

PROFILE_DIR = settings.PROJECT_ROOT / "config" / "profiles"

# 环境/命令行怎么指定档案名
ENV_PROFILE = "AI_AGENT_PROFILE"
DEFAULT_PROFILE = "ecommerce"

# 自动发现时要忽略的页面文件（基类、工具基类等）
_IGNORE_PAGE_FILES = {"__init__.py", "base_page.py", "common_page.py"}


@dataclass
class PageSpec:
    """一个页面对象的说明：它是谁、在哪、大概能干什么。"""

    keyword: str = ""          # 中文关键词，用于需求匹配（可为空=自动发现）
    class_name: str = ""       # 类名，如 LoginPage
    module: str = ""           # 模块路径，如 pages.login_page
    hint: str = ""             # 给模型看的方法提示（可为空，自动补真实方法）


@dataclass
class ProjectProfile:
    """一份"目标项目档案"：接哪个项目、怎么跑它、它有哪些页面对象。"""

    name: str = "unnamed"
    display_name: str = ""
    site_url: str = ""

    project_dir: Path = field(default_factory=Path)
    python_exe: Path = field(default_factory=Path)

    # 目标项目内部的相对结构（不同项目可不同）
    pages_dir: str = "pages"
    base_page_file: str = "base_page.py"
    conftest_file: str = "conftest.py"

    # 测试代码里怎么拿首页地址、第一个固件叫什么
    entry_snippet: str = ""
    page_fixture: str = "page_context"
    config_snippet: str = 'config["base_url"]'

    pages: list[PageSpec] = field(default_factory=list)
    fallback_page: PageSpec | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # 保证是 Path 类型（YAML 读出来是字符串）
        self.project_dir = Path(self.project_dir)
        self.python_exe = Path(self.python_exe)

    @property
    def pages_root(self) -> Path:
        """目标项目的 pages/ 目录绝对路径。"""
        return self.project_dir / self.pages_dir

    @property
    def exists(self) -> bool:
        """目标项目目录是否真的在磁盘上。"""
        return self.project_dir.exists()

    # ------------------------------------------------------------------
    # 页面对象：优先用档案里的映射；没有就自动发现
    # ------------------------------------------------------------------
    def resolve_page(self, feature: str) -> PageSpec:
        """根据中文功能名，决定该用哪个页面对象。

        顺序：
            1. 档案里配了 pages 映射 → 关键词匹配（精确、省 token）
            2. 关键词没命中 / 根本没配 → 自动发现（通用）
            3. 自动发现也失败 → 返回一个兜底空壳，让调用方优雅降级
        """
        text = (feature or "").strip()

        # 1) 关键词匹配
        for spec in self.pages:
            if spec.keyword and spec.keyword in text:
                return spec

        # 2) 自动发现：从目标项目 pages/ 里找"名字相关"的类（如需求含 login）
        #    注意：只拿类名词干去**功能描述**里比对。
        #    千万别拿它去比 module 名 —— 模块名必然包含类名词干
        #    （pages.login_page 必然含 login），那样每条都会"命中"，
        #    弱匹配就退化成"永远返回第一个"，反而比不用更糟。
        discovered = self.discover_pages()
        if discovered:
            lowered = text.lower()
            for spec in discovered:
                stem = spec.class_name.replace("Page", "").lower()
                if stem and (stem in lowered or lowered in stem):
                    return spec

        # 3) 档案里声明了兜底页 → 用它（比"随便挑第一个"可控得多）
        #    注意顺序：兜底页优先于"自动发现的第一个"，
        #    否则会出现需求说"登录"、却匹配到 BaiduPage 这种荒唐结果。
        if self.fallback_page:
            logger.info("功能『%s』未匹配到页面，使用档案声明的兜底页 %s",
                        text, self.fallback_page.class_name)
            return self.fallback_page

        # 4) 最后手段：自动发现的第一个
        if discovered:
            logger.info(
                "功能『%s』未匹配到页面，自动发现的 %d 个页面中取第一个：%s",
                text, len(discovered), discovered[0].class_name,
            )
            return discovered[0]

        return PageSpec()

    def discover_pages(self) -> list[PageSpec]:
        """扫描目标项目的 pages/ 目录，自动列出所有页面类和它们的真实方法。

        用 ast 静态解析，**不 import、不执行**目标项目代码（和 Day 18 同一套安全做法）。
        """
        root = self.pages_root
        if not root.exists():
            logger.warning("目标项目 pages 目录不存在：%s", root)
            return []

        specs: list[PageSpec] = []
        for path in sorted(root.glob("*.py")):
            if path.name in _IGNORE_PAGE_FILES or path.name.startswith("_"):
                continue
            classes = _class_methods(path)
            for class_name, methods in classes.items():
                if not class_name.endswith("Page"):
                    continue      # 只认 *Page 结尾的类，避免误抓辅助类
                module = f"{self.pages_dir}.{path.stem}"
                hint = _format_methods(methods)
                specs.append(PageSpec(
                    keyword="", class_name=class_name,
                    module=module, hint=hint,
                ))
        return specs

    def load_methods(self, module: str) -> set[str]:
        """拿到某个页面模块的真实方法名（含基类继承来的）。

        这是防"模型编方法名"的核心检查（Day 18），现在从档案驱动。
        """
        page_path = self.pages_root / Path(module.replace(".", "/") + ".py").name

        methods: set[str] = set()
        for names in _class_methods(page_path).values():
            methods |= names

        # 加上基类提供的方法（如 open / click / fill / get_text）
        base_path = self.pages_root / self.base_page_file
        for names in _class_methods(base_path).values():
            methods |= names

        return methods

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """给日志/命令行看的一行摘要。"""
        pages = self.pages or self.discover_pages()
        return (
            f"档案[{self.name}] {self.display_name or ''} ｜ "
            f"项目={self.project_dir} ｜ 页面对象 {len(pages)} 个"
            + ("" if self.exists else " ｜ ⚠️ 项目目录不存在")
        )


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
def load_profile(name: str | None = None) -> ProjectProfile:
    """按名字加载档案；找不到就返回一个基于环境变量/默认值的档案。"""
    name = name or os.getenv(ENV_PROFILE, DEFAULT_PROFILE)
    path = PROFILE_DIR / f"{name}.yaml"

    if not path.exists():
        logger.warning("找不到档案 %s（%s），改用内置默认配置", name, path)
        return _default_profile()

    try:
        data = file_io.read_yaml(path)
    except Exception as exc:  # noqa: BLE001 - 档案读坏不能让整个程序起不来
        logger.error("档案 %s 读取失败：%s，改用内置默认配置", path, exc)
        return _default_profile()

    return _from_dict(name, data)


def available_profiles() -> list[str]:
    """列出 config/profiles/ 下所有可用档案名。"""
    if not PROFILE_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def _default_profile() -> ProjectProfile:
    """没有档案文件时的兜底：直接用 settings 里的目标项目配置。"""
    return ProjectProfile(
        name="default",
        display_name="（未配置档案，沿用 settings）",
        project_dir=settings.TEST_PROJECT_DIR,
        python_exe=settings.TEST_PROJECT_PYTHON,
    )


def _from_dict(name: str, data: dict[str, Any]) -> ProjectProfile:
    """把 YAML 字典转成 ProjectProfile。字段缺失就用默认值，不报错。"""
    target = data.get("target") or {}
    project_dir = Path(
        target.get("project_dir")
        or os.getenv("TEST_PROJECT_DIR")
        or str(settings.TEST_PROJECT_DIR)
    )
    # python_exe 允许写相对路径（相对 project_dir），也允许绝对路径
    exe_raw = target.get("python_exe") or "venv/Scripts/python.exe"
    exe = Path(exe_raw)
    if not exe.is_absolute():
        exe = project_dir / exe

    pages: list[PageSpec] = []
    for item in data.get("pages") or []:
        if not isinstance(item, dict):
            continue
        pages.append(PageSpec(
            keyword=str(item.get("keyword", "")),
            class_name=str(item.get("class", "")),
            module=str(item.get("module", "")),
            hint=str(item.get("hint", "")),
        ))

    fallback = None
    fb = data.get("fallback_page")
    if isinstance(fb, dict):
        fallback = PageSpec(
            keyword=str(fb.get("keyword", "")),
            class_name=str(fb.get("class", "")),
            module=str(fb.get("module", "")),
            hint=str(fb.get("hint", "")),
        )

    return ProjectProfile(
        name=name,
        display_name=str(data.get("display_name", "")),
        site_url=str(data.get("site_url", "")),
        project_dir=project_dir,
        python_exe=exe,
        pages_dir=str(target.get("pages_dir", "pages")),
        base_page_file=str(target.get("base_page_file", "base_page.py")),
        conftest_file=str(target.get("conftest_file", "conftest.py")),
        entry_snippet=str(data.get("entry_snippet", "")),
        page_fixture=str(data.get("page_fixture", "page_context")),
        config_snippet=str(data.get("config_snippet", 'config["base_url"]')),
        pages=pages,
        fallback_page=fallback,
    )


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _class_methods(path: Path) -> dict[str, set[str]]:
    """静态解析一个 .py 文件，返回 {类名: 方法名集合}。

    和 agent/code_generator.py 里的同名函数逻辑一致 —— 那里会逐渐改为调用本函数，
    避免出现两套解析规则（两处真值不同步正是这次重构要根治的毛病）。
    """
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return {}

    classes: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names = {child.name for child in node.body
                     if isinstance(child, ast.FunctionDef)}
            classes[node.name] = names
    return classes


def _format_methods(methods: set[str], limit: int = 8) -> str:
    """把方法集合格式化成给模型看的提示文本（太长就截断）。"""
    if not methods:
        return ""
    ordered = sorted(m for m in methods if not m.startswith("_"))
    shown = ordered[:limit]
    text = " / ".join(f"{m}()" for m in shown)
    if len(ordered) > limit:
        text += f" 等 {len(ordered)} 个方法"
    return text
