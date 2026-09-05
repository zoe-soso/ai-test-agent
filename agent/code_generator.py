"""
测试用例 -> Playwright 代码（Day 17）

前面 16 天我们一直在生成"人看的东西"：结构化的用例、数据、评审报告。
从今天开始，AI 的产物变成**代码**——能被 pytest 真正执行的自动化测试脚本。

    测试用例（YAML/JSON）
          │
          ▼
      大模型 + 框架规范卡
          │
          ▼
      Playwright 测试代码
          │
          ▼
    代码检查（能不能跑、守不守规矩）
          │
          ▼
      保存到 generated_tests/

------------------------------------------------------------------
为什么生成代码前必须先"定规矩"（Day 16 的意义）
------------------------------------------------------------------
大模型什么代码都写得出来，但它不知道**你们团队**的规矩。
你不说清楚，它就会写成这样：

    def test_login(page):
        page.goto("https://...")
        page.fill("input[data-qa='login-email']", "a@b.com")   # 裸 API
        page.click("button[data-qa='login-button']")

这能跑，但它**不符合你的 POM 框架**：定位器散落在测试里，页面一改版要改几十个地方。
所以 Day 16 我们把规矩写成了 `prompts/code_generation.txt`，
把"必须调用页面对象方法、禁止裸 API、数据从 YAML 读"这些约束**钉死在 prompt 里**。

------------------------------------------------------------------
为什么还要用 Python 再检查一遍（code check）
------------------------------------------------------------------
prompt 里的规矩是"请求"，不是"保证"。大模型偶尔会不听话。
所以生成完必须用代码再验一遍——就像 Day 11 评审用例一样：
**AI 生成的东西，必须经过一次自动检查才能用。** 这是本项目一贯的思路。

检查分两类：
    1. 语法层面：这代码本身能不能被 Python 解析（用 ast 模块，不用真的执行）
    2. 规范层面：守没守 POM、有没有导入页面对象、有没有用 time.sleep 等
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import llm_client, project_profile
from agent.models import TestCase
from config import settings
from prompts import loader
from tools import file_io
from tools.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATE = "code_generation"
FIX_TEMPLATE = "fix_code"

# ----------------------------------------------------------------------
# 功能 -> 页面对象 的对应表
# ----------------------------------------------------------------------
# 为什么要这张表？
#   用例里只写了"用户登录功能"这种中文，代码里却要知道该 import 哪个类。
#   这张表就是"中文需求"和"代码里的类"之间的翻译字典。
#   以后新增页面，只在这里加一行。
#
# 格式：关键字 -> (页面类名, 模块路径, 给模型看的方法提示)
PAGE_REGISTRY: list[tuple[str, str, str, str]] = [
    (
        "登录",
        "LoginPage",
        "pages.login_page",
        "open_login_page() / login(email, password) / get_login_user() / "
        "get_error_message() / is_login_success(username) / logout()",
    ),
    (
        "购物车",
        "CartPage",
        "pages.cart_page",
        "打开购物车、添加商品、查看商品数量、删除商品等（以实际类方法为准）",
    ),
    (
        "结算",
        "CheckoutPage",
        "pages.checkout_page",
        "填写地址、下单、确认订单等（以实际类方法为准）",
    ),
    (
        "商品",
        "ProductPage",
        "pages.product_page",
        "搜索商品、打开详情、加入购物车等（以实际类方法为准）",
    ),
    (
        "注册",
        "LoginPage",
        "pages.login_page",
        "signup(name, email) / fill_account_info(info) / create_account() / "
        "is_account_created()",
    ),
]

# 兜底：认不出来就当登录页（本项目目前主要做登录）
FALLBACK_PAGE = ("LoginPage", "pages.login_page", "login(email, password) / get_login_user()")

# 测试代码里**不允许**出现的裸 Playwright API。
# 出现就说明模型没遵守 POM，把定位器写进测试里了。
FORBIDDEN_IN_TEST = (
    "page.locator(",
    "page.click(",
    "page.fill(",
    "page.goto(",
)


def resolve_page(
    feature: str,
    profile: project_profile.ProjectProfile | None = None,
) -> tuple[str, str, str]:
    """根据功能名找到该用哪个页面对象。返回 (类名, 模块, 方法提示)。

    Day 28 重构：不再只查本文件里写死的 PAGE_REGISTRY，
    而是优先走"项目档案"（config/profiles/*.yaml）。
    档案支持自动发现目标项目的页面类，所以接新项目**不用改这里的代码**。

    找不到档案时，退回旧的 PAGE_REGISTRY 行为（向后兼容，保证老测试仍能跑）。
    """
    prof = profile if profile is not None else project_profile.load_profile()
    if prof.pages or prof.fallback_page or prof.exists:
        spec = prof.resolve_page(feature)
        if spec.class_name:
            return spec.class_name, spec.module, spec.hint

    # 兜底：沿用本文件里的硬编码表（老行为）
    for keyword, page_class, module, hint in PAGE_REGISTRY:
        if keyword in feature:
            return page_class, module, hint
    return FALLBACK_PAGE


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class GeneratedCode:
    """一次代码生成的完整结果。"""

    feature: str
    case_id: str
    case_name: str
    code: str = ""
    filename: str = ""
    issues: list[str] = field(default_factory=list)
    attempts: int = 0          # 调了几次大模型
    repairs_used: int = 0      # 自修正了几次
    path: str = ""             # 保存后的路径

    @property
    def ok(self) -> bool:
        """是否合格：有代码 + 没查出问题。"""
        return bool(self.code) and not self.issues

    def describe(self) -> str:
        if self.ok:
            return f"{self.case_id} 代码生成合格（{len(self.code.splitlines())} 行）"
        if not self.code:
            return f"{self.case_id} 未生成出代码"
        return f"{self.case_id} 有 {len(self.issues)} 处问题"


# ----------------------------------------------------------------------
# 代码检查（Day 18 的核心：生成完先自己验一遍）
# ----------------------------------------------------------------------
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_python_code(raw: str) -> str:
    """从大模型的回复里把 ```python ... ``` 代码块抠出来。

    为什么需要这个函数？
        你要求模型"只输出代码块"，它还是经常在前后加一句
        "好的，这是你要的代码："。别跟它较真，用正则把代码块抠出来就行。
        这和 Day 7 去掉 JSON 围栏是同一类问题。
    """
    matched = _CODE_BLOCK.search(raw)
    if matched:
        return matched.group(1).strip()
    # 没写围栏：整段当代码（后面 validate 如果解析不了会报错）
    return raw.strip()


# ----------------------------------------------------------------------
# 静态读取目标项目的页面对象（拿到"真实存在的方法"清单）
# ----------------------------------------------------------------------
def _class_methods(path: Path) -> dict[str, set[str]]:
    """解析一个 .py 文件，返回 {类名: 该类定义的方法名集合}。

    用 ast 解析源码而不是 import 这个模块，有两个原因：
        1. 页面对象依赖 playwright，而我们项目没装（两个项目环境隔离）
        2. 只读源码，不会执行目标项目的任何代码，绝对安全
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
            names = {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
            classes[node.name] = names
    return classes


def load_page_methods(
    page_module: str = "pages.login_page",
    profile: project_profile.ProjectProfile | None = None,
) -> set[str]:
    """拿到某个页面对象**真实拥有**的方法名（含从 BasePage 继承来的）。

    这是本项目最实用的一道检查。为什么必须有它？

        大模型写测试代码时，最爱犯的错就是**编造方法名**。
        它会一本正经地写出 login_page.fill_email(...) 这种看起来很合理的调用，
        但你的 LoginPage 里根本没有 fill_email —— 只有 login(email, password)。
        这种代码语法完全正确、POM 规范也守了，一跑就 AttributeError。

        所以光检查"守不守规矩"不够，还得检查"调的方法是不是真的存在"。
        方法清单直接从目标项目源码里静态解析，永远和真实代码同步。
    """
    # Day 28：改走"项目档案"，路径和基类文件名都从档案读，
    # 这样接一个 pages 目录结构不同的项目也不用改代码。
    prof = profile if profile is not None else project_profile.load_profile()
    return prof.load_methods(page_module)


def _find_page_calls(tree: ast.AST, page_class: str) -> tuple[str | None, list[str]]:
    """找出代码里"页面对象变量都调了哪些方法"。

    做法：
        1. 先找 `xxx = LoginPage(page_context)` 这种赋值，记下变量名 xxx
        2. 再找所有 `xxx.method(...)` 调用，收集 method 名字
    """
    page_var: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = getattr(func, "id", None) or getattr(func, "attr", None)
            if func_name == page_class and node.targets:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    page_var = target.id

    called: list[str] = []
    if page_var:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == page_var
            ):
                called.append(node.func.attr)

    return page_var, called


def validate_code(
    code: str,
    page_class: str = "LoginPage",
    page_module: str = "pages.login_page",
    profile: project_profile.ProjectProfile | None = None,
) -> list[str]:
    """检查生成的代码，返回问题清单（空列表 = 合格）。

    注意：这里**不执行**代码，只用 ast 做静态检查。
    执行未知代码是危险的，这也是 Day 19 要加人工确认的原因。

    Day 28：方法真实性检查改为从"项目档案"读，可适配任意目标项目。
    """
    issues: list[str] = []

    # ---- 1. 语法检查：这段代码本身是不是合法 Python ----
    # ast.parse 只"读懂"代码结构，不会运行它，所以是安全的。
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Python 语法错误（第 {exc.lineno} 行）：{exc.msg}"]

    if not code.strip():
        return ["代码是空的"]

    # ---- 2. 有没有导入对应的页面对象 ----
    expected_import = f"from {page_module} import"
    if expected_import not in code:
        issues.append(
            f"没有导入页面对象，应包含：{expected_import} {page_class}"
        )

    # ---- 3. 有没有违反 POM（在测试里直接用裸 API）----
    for bad in FORBIDDEN_IN_TEST:
        if bad in code:
            issues.append(
                f"违反 POM 规范：测试代码里出现裸 Playwright API「{bad}」，"
                f"应改成调用 {page_class} 的方法"
            )

    # ---- 4. 有没有测试函数、函数参数里有没有 page_context 固件 ----
    test_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if not test_funcs:
        issues.append("没有找到测试函数（函数名要以 test_ 开头）")
    else:
        for func in test_funcs:
            params = [arg.arg for arg in func.args.args]
            if "page_context" not in params:
                issues.append(
                    f"测试函数 {func.name} 的参数里缺少 page_context 固件"
                    f"（当前参数：{params or '无'}）"
                )

    # ---- 5. 方法真实性检查：调的方法在页面对象里真的存在吗？ ----
    # 这是抓"模型编造方法名"的关键一道关（详见 load_page_methods 的注释）。
    real_methods = load_page_methods(page_module, profile)
    if real_methods:
        _page_var, called_methods = _find_page_calls(tree, page_class)
        unknown = sorted({m for m in called_methods if m not in real_methods})
        if unknown:
            available = "、".join(sorted(real_methods))
            issues.append(
                f"调用了 {page_class} 上不存在的方法：{'、'.join(unknown)}。"
                f"该类真实可用的方法只有：{available}"
            )
        elif not called_methods:
            issues.append(f"没有看到任何通过 {page_class} 调用的方法，测试可能是空的")

    # ---- 6. 有没有先打开被测站点 ----
    # 页面对象里的 open_login_page() 只是点击页面上的"登录"链接，
    # 前提是浏览器已经在站点首页上。不先 open 首页，后面的点击全部会超时。
    if ".open(" not in code and "base_url" not in code:
        issues.append(
            "测试没有先打开被测站点首页（应先调用 open(站点地址) 或读取配置里的 base_url）"
        )

    # ---- 7. 其他坏习惯 ----
    if "time.sleep" in code:
        issues.append("不要用 time.sleep 等待，页面对象内部已有等待机制")

    return issues


# ----------------------------------------------------------------------
# 生成器
# ----------------------------------------------------------------------
class CodeGenerator:
    """把测试用例转成 Playwright 测试代码。"""

    def __init__(
        self,
        client: llm_client.LLMClient | None = None,
        template: str = DEFAULT_TEMPLATE,
        max_repairs: int = 1,
        profile: project_profile.ProjectProfile | None = None,
    ) -> None:
        self.client = client or llm_client.get_client()
        self.template = template
        self.max_repairs = max_repairs
        # Day 28：一份档案决定"接哪个项目、用哪个页面对象"。
        # 只加载一次，后面到处复用（避免每个用例都去读一遍 YAML + 扫源码）。
        self.profile = profile if profile is not None else project_profile.load_profile()

    # ------------------------------------------------------------------
    def generate(self, feature: str, case: TestCase) -> GeneratedCode:
        """为一个测试用例生成测试代码（含自修正）。"""
        page_class, page_module, page_hint = resolve_page(feature, self.profile)

        result = GeneratedCode(
            feature=feature,
            case_id=str(case.get("id", "?")),
            case_name=str(case.get("name", "")),
        )

        # 把用例转成精简 JSON 发给模型（只发需要的字段，省 token）
        case_json = _render_case(case)

        # ---- 1. 生成 ----
        raw = self._call(feature, case_json, page_class, page_module, page_hint)
        result.attempts = 1

        code = extract_python_code(raw)
        issues = validate_code(code, page_class, page_module, self.profile)

        # ---- 2. 自修正：把问题清单告诉模型，让它改一版 ----
        while issues and result.repairs_used < self.max_repairs:
            result.repairs_used += 1
            result.attempts += 1
            logger.info(
                "%s 代码检查发现 %d 处问题，启动自修正", result.case_id, len(issues)
            )
            fix_prompt = loader.load(FIX_TEMPLATE, errors="\n".join(f"- {i}" for i in issues), raw=raw)
            raw = self.client.chat_messages(fix_prompt.to_messages(), mock_hint="code")
            code = extract_python_code(raw)
            issues = validate_code(code, page_class, page_module, self.profile)

        result.code = code
        result.issues = issues
        result.filename = _make_filename(case)
        logger.info("代码生成完成：%s", result.describe())
        return result

    def generate_many(self, feature: str, cases: list[TestCase], limit: int = 1) -> list[GeneratedCode]:
        """批量生成（默认只生成第 1 条，省钱）。"""
        return [self.generate(feature, case) for case in cases[:max(1, limit)]]

    # ------------------------------------------------------------------
    def save(self, result: GeneratedCode, filename: str | None = None) -> Path:
        """把代码保存到 generated_tests/ 目录。"""
        name = filename or result.filename
        if not name.endswith(".py"):
            name += ".py"
        target = settings.GENERATED_DIR / name
        file_io.write_text(target, result.code)
        result.path = str(target)
        logger.info("测试代码已保存到：%s", target)
        return target

    # ------------------------------------------------------------------
    def _call(
        self,
        feature: str,
        case_json: str,
        page_class: str,
        page_module: str,
        page_hint: str,
    ) -> str:
        # 把"真实存在的方法清单"直接告诉模型。
        #
        # 这是防止模型编造方法名最有效的一招：
        # 与其等它写错了再用检查去纠错，不如**一开始就给它准确的清单**。
        # （纠错要花第二次调用的钱，而且它第二次还可能编出新的错名字。）
        real_methods = load_page_methods(page_module, self.profile)
        page_methods = "、".join(sorted(m for m in real_methods if not m.startswith("__")))

        prompt = loader.load(
            self.template,
            feature=feature,
            case=case_json,
            page_class=page_class,
            page_module=page_module,
            page_hint=page_hint,
            page_methods=page_methods or "（未能读取到方法清单，请只使用 $page_hint 中列出的方法）".replace("$page_hint", page_hint),
        )
        return self.client.chat_messages(prompt.to_messages(), mock_hint="code")


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _render_case(case: TestCase) -> str:
    """把一条用例渲染成精简 JSON 字符串给模型看。"""
    import json

    slim = {
        "id": case.get("id"),
        "name": case.get("name"),
        "case_type": case.get("case_type"),
        "priority": case.get("priority"),
        "steps": case.get("steps", []),
        "expected": case.get("expected"),
    }
    return json.dumps(slim, ensure_ascii=False, indent=2)


def _make_filename(case: TestCase) -> str:
    """根据用例 ID 生成文件名，例如 TC_LOGIN_001 -> test_tc_login_001.py。"""
    raw = str(case.get("id", "case")).lower()
    safe = re.sub(r"[^a-z0-9_]", "_", raw)
    return f"test_{safe}.py"
