"""
离线 Mock 大模型（Day 6 起使用）

这个文件是**假的模型**，存在的意义是：没 API Key 时也能把整条链路跑通。

它比 Day 5 那个固定字符串聪明一点：**会模拟真实模型的四种坏毛病**。

    clean   规规矩矩返回 JSON
    fenced  返回 JSON，但包了一层 ```json 代码围栏
    chatty  在 JSON 前后加"好的，这是结果："和"希望对你有帮助"
    broken  返回的 JSON 缺字段、枚举值乱写

为什么故意让它犯病？
    因为 Day 7 要写的解析器、校验器、自修正重试，
    **只有在见过脏数据的时候才证明它真的有用**。
    如果你的 mock 永远返回完美 JSON，那套容错代码等于没测过。

    默认按顺序轮换这四种风格，所以连续跑几次就能看到四种表现，
    也能验证"重试一次就恢复"这条逻辑确实生效。
"""

from __future__ import annotations

import json
from typing import Any

# 轮换顺序：clean → fenced → chatty → broken → clean ...
STYLES: tuple[str, ...] = ("clean", "fenced", "chatty", "broken")

# 用来判断"这是不是一个要求返回 JSON 的 Prompt"
JSON_MARKERS = ("json", "JSON", "结构化")

# 用来判断"这是一个工程化设计的 Prompt"（而不是随手一句）
ENGINEERED_MARKERS = ("资深", "硬性约束", "等价类", "边界值", "case_type", "输出格式")


class MockLLM:
    """会轮换犯病的假模型。"""

    def __init__(self, style: str = "cycle") -> None:
        """
        style:
            "cycle" —— 按 clean/fenced/chatty/broken 轮换（默认）
            其他值  —— 固定用这种风格，写单元测试时用
        """
        self.style = style
        self._call_count = 0

    # ------------------------------------------------------------------
    def respond(self, prompt: str, system: str = "", hint: str | None = None) -> str:
        """根据 prompt 内容，返回一段"像模型会说的话"。

        hint 是调用方显式指定的回答类型（"naive" / "engineered" / "json"）。
        为什么要显式指定？
            因为靠猜 prompt 内容来判断"该返回哪种回答"太脆弱。
            调用方最清楚自己用的是哪个模板，让它直接说。
        """
        full = system + "\n" + prompt

        if hint == "json" or any(m in full for m in JSON_MARKERS):
            return self._json_response(self._next_style())

        if hint == "naive":
            return self._naive_response()

        if hint == "engineered" or any(m in full for m in ENGINEERED_MARKERS):
            return self._engineered_response()

        return self._generic_response()

    def _next_style(self) -> str:
        if self.style != "cycle":
            return self.style
        style = STYLES[self._call_count % len(STYLES)]
        self._call_count += 1
        return style

    # ------------------------------------------------------------------
    # 各种回答
    # ------------------------------------------------------------------
    def _json_response(self, style: str) -> str:
        """返回 JSON，按 style 决定脏不脏。"""
        cases = _json_cases(break_it=(style == "broken"))
        payload = {"feature": "用户登录功能", "cases": cases}
        raw = json.dumps(payload, ensure_ascii=False, indent=2)

        if style == "clean":
            return raw

        if style == "fenced":
            return f"```json\n{raw}\n```"

        if style == "chatty":
            return (
                "好的，我帮你设计了以下测试用例：\n\n"
                f"```json\n{raw}\n```\n\n"
                "如果还需要补充性能测试或兼容性测试的用例，随时告诉我。"
            )

        # broken：把 JSON 写坏，模拟"看起来像 JSON 但字段不对"
        return (
            "好的，这是结果：\n"
            "```json\n"
            + raw.replace('"expected":', '"expectd":', 1)  # 字段名拼错，校验会挂
            + "\n```"
        )

    def _engineered_response(self) -> str:
        """工程化 Prompt 的回答：8 条结构化用例。"""
        return (
            "【Mock：工程化 Prompt 的预设回答】\n\n"
            "1. id: TC_LOGIN_001\n"
            "   name: 使用已注册账号正常登录\n"
            "   case_type: 正常\n"
            "   priority: P0\n"
            "   steps:\n"
            "     - 打开首页\n"
            "     - 点击登录入口\n"
            "     - 输入已注册的邮箱\n"
            "     - 输入正确密码\n"
            "     - 点击登录按钮\n"
            "   expected: 登录成功，页面顶部显示 \"Logged in as 用户名\"\n\n"
            "2. id: TC_LOGIN_002\n"
            "   name: 密码错误时登录失败\n"
            "   case_type: 异常\n"
            "   priority: P1\n"
            "   steps:\n"
            "     - 打开首页\n"
            "     - 点击登录入口\n"
            "     - 输入已注册的邮箱\n"
            "     - 输入错误密码\n"
            "     - 点击登录按钮\n"
            "   expected: 页面提示 \"Your email or password is incorrect!\"\n\n"
            "（省略 6 条：账号不存在、邮箱为空、密码为空、"
            "密码 1 位、密码 500 位、邮箱含特殊字符）\n"
        )

    def _naive_response(self) -> str:
        """朴素 Prompt 的回答：只有 3 条，而且含糊、不可断言。"""
        return (
            "【Mock：朴素 Prompt 的预设回答】\n\n"
            "登录功能的测试用例如下：\n\n"
            "1. 正常登录\n"
            "   输入正确的用户名和密码，点击登录，验证能正常登录。\n\n"
            "2. 错误密码\n"
            "   输入错误的密码，验证系统会给出提示。\n\n"
            "3. 异常情况\n"
            "   测试一些异常输入，验证系统不会崩溃。\n\n"
            "（对比要点：只有 3 条、没有边界场景、"
            "预期结果写的是\"能正常登录\"\"不会崩溃\"，完全无法转成断言）\n"
        )

    def _generic_response(self) -> str:
        """Day 5 用的通用回答：直接问模型一句话时的样子。"""
        return (
            "【Mock 模式：以下内容是本地预设的假回答，未真实调用大模型】\n\n"
            "针对「用户登录功能」，我建议设计以下 8 条测试用例：\n\n"
            "一、正常场景\n"
            "1. TC_LOGIN_001 使用已注册账号正常登录（P0）\n"
            "   步骤：打开首页 → 点击登录入口 → 输入正确邮箱 → 输入正确密码 → 点击登录\n"
            "   预期：登录成功，顶部显示 Logged in as 用户名\n\n"
            "二、异常场景\n"
            "2. TC_LOGIN_002 密码错误（P1）→ 提示邮箱或密码错误\n"
            "3. TC_LOGIN_003 账号不存在（P1）→ 提示邮箱或密码错误\n"
            "4. TC_LOGIN_004 邮箱为空（P1）→ 不允许提交\n"
            "5. TC_LOGIN_005 密码为空（P2）→ 不允许提交\n\n"
            "三、边界场景\n"
            "6. TC_LOGIN_006 密码 1 位（P2）→ 登录失败但不崩溃\n"
            "7. TC_LOGIN_007 密码 500 位（P2）→ 有长度校验，页面不报错\n"
            "8. TC_LOGIN_008 邮箱含特殊字符（P3）→ 提示格式错误\n"
        )


# ----------------------------------------------------------------------
# JSON 用例数据
# ----------------------------------------------------------------------
def _json_cases(break_it: bool = False) -> list[dict[str, Any]]:
    """生成符合 TestCase 契约的 JSON 用例。

    break_it=True 时故意掺入两条脏数据，用来验证校验器：
        - 一条少了 expected 字段
        - 一条 case_type / priority 写成契约里没有的值
    """
    cases: list[dict[str, Any]] = [
        {
            "id": "TC_LOGIN_001",
            "name": "使用已注册账号正常登录",
            "case_type": "正常",
            "priority": "P0",
            "steps": ["打开首页", "点击登录入口", "输入已注册邮箱", "输入正确密码", "点击登录按钮"],
            "expected": "登录成功，页面顶部显示 Logged in as 用户名",
        },
        {
            "id": "TC_LOGIN_002",
            "name": "密码错误时登录失败",
            "case_type": "异常",
            "priority": "P1",
            "steps": ["打开首页", "点击登录入口", "输入已注册邮箱", "输入错误密码", "点击登录按钮"],
            "expected": "页面提示 Your email or password is incorrect!",
        },
        {
            "id": "TC_LOGIN_003",
            "name": "账号不存在时登录失败",
            "case_type": "异常",
            "priority": "P1",
            "steps": ["打开首页", "点击登录入口", "输入未注册邮箱", "输入任意密码", "点击登录按钮"],
            "expected": "登录失败，提示邮箱或密码错误",
        },
        {
            "id": "TC_LOGIN_004",
            "name": "邮箱为空时提交被拦截",
            "case_type": "异常",
            "priority": "P1",
            "steps": ["打开首页", "点击登录入口", "邮箱留空", "输入密码", "点击登录按钮"],
            "expected": "无法提交或提示邮箱必填",
        },
        {
            "id": "TC_LOGIN_005",
            "name": "密码为空时提交被拦截",
            "case_type": "异常",
            "priority": "P2",
            "steps": ["打开首页", "点击登录入口", "输入邮箱", "密码留空", "点击登录按钮"],
            "expected": "无法提交或提示密码必填",
        },
        {
            "id": "TC_LOGIN_006",
            "name": "密码取最小边界值",
            "case_type": "边界",
            "priority": "P2",
            "steps": ["打开首页", "点击登录入口", "输入正确邮箱", "输入1位密码", "点击登录按钮"],
            "expected": "登录失败，提示明确，页面不崩溃",
        },
        {
            "id": "TC_LOGIN_007",
            "name": "超长密码不导致系统异常",
            "case_type": "边界",
            "priority": "P2",
            "steps": ["打开首页", "点击登录入口", "输入正确邮箱", "输入500位密码", "点击登录按钮"],
            "expected": "登录失败，存在长度校验，无 500 错误",
        },
    ]

    if break_it:
        cases.append(
            {  # 缺 expected 字段
                "id": "TC_LOGIN_008",
                "name": "邮箱含特殊字符",
                "case_type": "边界",
                "priority": "P3",
                "steps": ["打开首页", "点击登录入口", "输入含特殊字符的邮箱", "点击登录按钮"],
            }
        )
        cases.append(
            {  # 枚举值越界
                "id": "TC_LOGIN_009",
                "name": "并发登录压力测试",
                "case_type": "性能测试",
                "priority": "P9",
                "steps": ["并发 100 个请求"],
                "expected": "系统不崩溃",
            }
        )
    else:
        cases.append(
            {
                "id": "TC_LOGIN_008",
                "name": "邮箱含特殊字符时不崩溃",
                "case_type": "边界",
                "priority": "P3",
                "steps": ["打开首页", "点击登录入口", "输入含特殊字符的邮箱", "输入密码", "点击登录按钮"],
                "expected": "提示格式错误，不发生 500 错误",
            }
        )

    # Day 10：给每条用例标注它用的设计方法。
    # 五种方法都安排上，这样"设计方法覆盖率"这个指标才有东西可看。
    methods = (
        "场景法", "判定表", "等价类划分", "异常测试",
        "异常测试", "边界值分析", "边界值分析", "异常测试",
    )
    for index, case in enumerate(cases):
        if index < len(methods):
            case["design_method"] = methods[index]

    return cases
