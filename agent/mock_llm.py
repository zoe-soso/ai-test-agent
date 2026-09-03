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

from agent.tool_calling import AssistantMessage, ToolCall

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
        # 工具调用剧本走到第几步了（Day 13）。
        # 它是实例状态而不是局部变量，因为一次对话要跨好几轮循环，
        # 每一轮都是新的方法调用，得靠它记住"我上一轮已经调过什么了"。
        self._tool_step = 0

    # ------------------------------------------------------------------
    def respond(self, prompt: str, system: str = "", hint: str | None = None) -> str:
        """根据 prompt 内容，返回一段"像模型会说的话"。

        hint 是调用方显式指定的回答类型（"naive" / "engineered" / "json"）。
        为什么要显式指定？
            因为靠猜 prompt 内容来判断"该返回哪种回答"太脆弱。
            调用方最清楚自己用的是哪个模板，让它直接说。
        """
        full = system + "\n" + prompt

        # Day 11：评审请求要返回"评审结论"结构，不是用例列表。
        # 如果这里不做区分，mock 会返回 {"cases": [...]}，
        # review_by_llm 解析后取不到 overall/issues，评审链路在离线模式下
        # 就永远不会真正跑到 —— 等于这段容错代码没被测过。
        if hint == "review":
            return self._review_response(self._next_style())

        # Day 12：测试数据的结构（params + data）和用例（cases）不同，
        # 同样必须区分，否则离线时数据链路取不到 fields，会被判成全部不合格。
        if hint == "testdata":
            return self._testdata_response(self._next_style())

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
    # Day 13：工具调用
    # ------------------------------------------------------------------
    def respond_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        fault: bool = False,
    ) -> AssistantMessage:
        """离线模拟"模型决定调用工具"这一步。

        为什么 Mock 也要支持工具调用？
            理由和上面一模一样：**没被离线跑过的代码等于没测过**。
            `run_tool_loop` 里的"追加 assistant 消息 → 执行 → 回传结果"
            这段逻辑，如果只有真实调用才走到，
            那没 Key 的人（以及 CI）就永远验证不了它。

        它按剧本走：看用户最后一句话里有什么词，决定调哪个工具，
        调完一次之后就不再调，改成给最终答复。

        fault=True 时会**故意**先调一个不存在的工具。
        这不是捣乱，是为了验证：工具报错被当成结果回传给模型之后，
        模型能不能自己纠正过来。真实模型确实经常传错工具名或参数，
        这条纠错路径如果不提前测过，上线就是事故。
        """
        self._tool_step += 1
        user_text = _last_user_text(messages)

        # 第 1 轮：先来一次错误调用（仅在 fault 模式），看程序能不能扛住
        if fault and self._tool_step == 1:
            return AssistantMessage(
                content="我先把旧的用例文件删掉，免得混淆。",
                tool_calls=[ToolCall(
                    id=f"call_{self._tool_step}",
                    name="delete_file",
                    arguments={"path": "outputs/testcases.yaml"},
                )],
            )

        plan = _tool_plan(user_text, fault=fault)
        step_index = self._tool_step - (2 if fault else 1)

        if step_index < len(plan["calls"]):
            name, arguments = plan["calls"][step_index]
            return AssistantMessage(
                content=plan["say"][step_index] if step_index < len(plan["say"]) else None,
                tool_calls=[ToolCall(
                    id=f"call_{self._tool_step}",
                    name=name,
                    arguments=arguments,
                )],
            )

        return AssistantMessage(content=plan["final"])

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

    def _review_response(self, style: str) -> str:
        """Day 11：返回"评审结论"JSON（结构和用例列表完全不同）。

        内容刻意设计成**会触发修改**的形态：
        包含 1 个错误级 + 2 个警告级问题、2 个遗漏场景。
        这样 `review_and_revise` 的"修改"分支在离线模式下也能真正跑到，
        否则那段代码永远只在真实调用时才被执行 —— 太晚了。
        """
        raw = json.dumps(_review_data(), ensure_ascii=False, indent=2)

        if style == "clean":
            return raw

        if style == "fenced":
            return f"```json\n{raw}\n```"

        if style == "chatty":
            return (
                "好的，我评审完了，结论如下：\n\n"
                f"```json\n{raw}\n```\n\n"
                "如果你需要我直接把用例改好，可以再告诉我一声。"
            )

        # broken：把 issues 里的字段名拼错，模拟"看着像 JSON 但字段不对"
        return (
            "好的，这是评审结果：\n"
            "```json\n"
            + raw.replace('"problem":', '"problm":', 1)
            + "\n```"
        )

    def _testdata_response(self, style: str) -> str:
        """Day 12：返回测试数据 JSON（结构是 params + data，不是 cases）。"""
        payload = _testdata_data(break_it=(style == "broken"))
        raw = json.dumps(payload, ensure_ascii=False, indent=2)

        if style == "clean":
            return raw

        if style == "fenced":
            return f"```json\n{raw}\n```"

        if style == "chatty":
            return (
                "好的，这是为登录功能设计的测试数据：\n\n"
                f"```json\n{raw}\n```\n\n"
                "如果需要补充更多边界数据，随时告诉我。"
            )

        # broken：把第一组的 fields 字段名拼错，校验会挂
        return (
            "好的，数据如下：\n"
            "```json\n"
            + raw.replace('"fields":', '"fieldz":', 1)
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
# 工具调用剧本（Day 13）
# ----------------------------------------------------------------------
def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """取对话里最后一句用户说的话。"""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _tool_plan(user_text: str) -> dict[str, Any]:
    """根据用户说了什么，决定该调哪些工具。

    关键词判断很粗糙，但够用 —— Mock 的目的是**走通链路**，
    不是真的理解语义。真要理解语义，那活儿归模型。
    """
    text = user_text.lower()

    if any(word in text for word in ("失败", "分析", "报错", "为什么", "挂了")):
        return {
            "calls": [
                ("run_pytest", {"target": "tests", "timeout": 300}),
                ("analyze_failure", {
                    "log_text": (
                        "FAILED tests/test_x.py::test_login - AssertionError: "
                        "assert 'Logged in as' in 'Your email or password is incorrect!'\n"
                        "E   AssertionError: 断言失败\n"
                        "1 failed, 42 passed in 3.21s"
                    )
                }),
            ],
            "say": [
                "我先跑一遍测试，看看现在的实际情况。",
                "有 1 条失败了，我来分析一下原因。",
            ],
            "final": (
                "跑完了：42 条通过，1 条失败。\n"
                "失败的是 test_login，类型是**断言失败** —— "
                "实际返回了「邮箱或密码错误」的提示，说明登录根本没成功。\n"
                "建议：先核对这条用例的预期结果和当前测试数据是否还匹配，"
                "确认无误再提缺陷单。"
            ),
        }

    if any(word in text for word in ("跑", "执行", "pytest", "回归")):
        return {
            "calls": [("run_pytest", {"target": "tests", "timeout": 300})],
            "say": ["好的，我来执行一遍测试。"],
            "final": "测试已执行完毕，结果如上。全部用例的通过/失败情况都列出来了。",
        }

    if any(word in text for word in ("读", "看", "有哪些", "列出", "目录", "文件")):
        return {
            "calls": [
                ("list_files", {"directory": "outputs", "pattern": "*.yaml"}),
                ("read_file", {"path": "outputs/testcases.yaml"}),
            ],
            "say": [
                "我先看看 outputs 目录下有哪些文件。",
                "我读一下用例文件的内容。",
            ],
            "final": (
                "outputs 目录里的 YAML 文件已经列出来了，"
                "用例文件的内容也读到了。需要我对这些用例做什么处理，"
                "比如评审质量或者补充边界场景吗？"
            ),
        }

    # 默认剧本：生成用例 -> 保存
    # 这是最能体现 Agent 价值的一条链：
    # 模型不是"说出"该怎么做，而是真的把用例生成出来并存成了文件。
    return {
        "calls": [
            ("generate_testcase", {
                "feature": "用户登录功能",
                "description": "邮箱+密码登录，支持记住我",
            }),
            ("save_yaml", {"filename": "testcases.yaml"}),
        ],
        "say": [
            "好的，我先为登录功能生成测试用例。",
            "用例生成好了，我把它保存到文件里。",
        ],
        "final": (
            "已经完成了：为「用户登录功能」生成了测试用例并保存到 "
            "outputs/testcases.yaml。\n\n"
            "说明：当前是 **Mock 离线模式**，用例内容来自本地预置数据，"
            "没有真的调用大模型。配置好 API Key 后同样的命令会走真实生成。"
        ),
    }


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


def _review_data() -> dict[str, Any]:
    """构造一份"评审结论"数据（Day 11）。

    字段名必须和 prompts/review.txt 里要求的输出 schema 严格一致：
        overall / missing_scenarios / issues[{case_id, category, problem,
                                             suggestion, severity}]

    评审结果**同样**是模型的输出，同样可能不合规 ——
    这一点很容易被忽略：大家只记得给"生成用例"加容错，
    忘了"评审""修改"这些环节的返回也需要同样的防御。
    """
    return {
        "overall": "整体可用，但异常场景覆盖不足，且有一条用例的预期结果过于含糊。",
        "missing_scenarios": [
            "连续多次登录失败后账号被锁定",
            "密码前后含空格时的处理",
        ],
        "issues": [
            {
                "case_id": "TC_LOGIN_006",
                "category": "预期可验证性",
                "problem": "预期结果「登录失败，提示明确」不够具体，执行时无法判断通过与否",
                "suggestion": "改成可验证的描述，如「提示密码长度至少 6 位」",
                "severity": "警告",
            },
            {
                "case_id": "TC_LOGIN_008",
                "category": "步骤合理性",
                "problem": "步骤里没有输入密码就直接点击登录按钮，缺少必要操作",
                "suggestion": "补上「输入密码」这一步",
                "severity": "错误",
            },
            {
                "case_id": "-",
                "category": "异常遗漏",
                "problem": "未覆盖连续登录失败导致账号锁定的场景",
                "suggestion": "新增一条异常用例覆盖账号锁定",
                "severity": "警告",
            },
        ],
    }


def _testdata_data(break_it: bool = False) -> dict[str, Any]:
    """构造登录功能的测试数据（Day 12）。

    六类数据全部覆盖，且刻意包含**真实**的内容：
        - 500 字符的超长密码（用紧凑标记 <<LONG:500:A>> 表示，
          由程序展开 —— 模型逐字输出长串既贵又不可靠，见 testdata_generator）
        - 真的 SQL 注入 / XSS 片段

    这样离线时"标记展开""占位符检测""超长长度检测"这几条规则都有东西可查 ——
    否则 mock 一直返回完美数据，那些检测代码等于从没跑过。
    """


    data: list[dict[str, Any]] = [
        {
            "id": "TD_LOGIN_001",
            "name": "已注册账号的正确凭证",
            "data_type": "正确数据",
            "fields": {"username": "testuser@example.com", "password": "Test@123456"},
            "purpose": "验证正常登录流程",
            "expected": "登录成功，页面显示 Logged in as testuser",
        },
        {
            "id": "TD_LOGIN_002",
            "name": "密码错误",
            "data_type": "错误数据",
            "fields": {"username": "testuser@example.com", "password": "WrongPass999"},
            "purpose": "验证密码校验逻辑",
            "expected": "登录失败，提示邮箱或密码错误",
        },
        {
            "id": "TD_LOGIN_003",
            "name": "密码留空",
            "data_type": "空值",
            "fields": {"username": "testuser@example.com", "password": ""},
            "purpose": "验证必填项校验",
            "expected": "无法提交或提示密码必填",
        },
        {
            "id": "TD_LOGIN_004",
            "name": "500 位超长密码",
            "data_type": "超长数据",
            "fields": {"username": "testuser@example.com", "password": "<<LONG:500:A>>"},
            "purpose": "验证超长输入不导致系统异常",
            "expected": "存在长度校验，页面不崩溃、无 500 错误",
        },
        {
            "id": "TD_LOGIN_005",
            "name": "SQL 注入片段",
            "data_type": "特殊字符",
            "fields": {"username": "' OR '1'='1", "password": "anything"},
            "purpose": "验证 SQL 注入防护",
            "expected": "登录失败，不发生注入，无数据库报错",
        },
        {
            "id": "TD_LOGIN_006",
            "name": "密码含 XSS 片段",
            "data_type": "特殊字符",
            "fields": {
                "username": "testuser@example.com",
                "password": "<script>alert(1)</script>",
            },
            "purpose": "验证 XSS 防护",
            "expected": "脚本不被执行，原样处理或提示格式错误",
        },
        {
            "id": "TD_LOGIN_007",
            "name": "未注册的账号",
            "data_type": "不存在数据",
            "fields": {
                "username": "notexist_9527@example.com",
                "password": "Test@123456",
            },
            "purpose": "验证不存在的账号无法登录",
            "expected": "登录失败，提示邮箱或密码错误",
        },
    ]

    if break_it:
        # 掺两条脏数据，用来验证测试数据校验器：
        #   一条是描述性占位符（能通过结构校验，但质量检查要能揪出来）
        #   一条缺 fields（结构校验就该拦下）
        data.append({
            "id": "TD_LOGIN_008",
            "name": "占位符数据",
            "data_type": "正确数据",
            "fields": {"username": "正确的用户名", "password": "正确的密码"},
            "purpose": "用于验证占位符检测",
        })
        data.append({
            "id": "TD_LOGIN_009",
            "name": "缺少 fields",
            "data_type": "空值",
            "purpose": "用于验证 fields 缺失会被丢弃",
        })

    return {"params": ["username", "password"], "data": data}
