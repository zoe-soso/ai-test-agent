"""
纯 Python 测试用例生成器（Day 2 —— 先不用 AI）

Day 2 的目的不是做出多牛的东西，而是：
    1. 复习 list / dict / tuple / set 四种数据结构
    2. 复习 def / return，把"功能"封装成函数
    3. 先把数据结构定下来，明天 Day 3 才能顺理成章地写成 YAML

为什么先写不用 AI 的版本？
    因为后面 AI 生成的用例，最终也要落成一模一样的数据结构。
    先手写一遍，你就真正理解了"AI 到底在替我生成什么"。
    直接跳到 AI，你只会 copy prompt，讲不出原理。

四种数据结构在这里各司其职：
    list  装用例序列（有序、可变）
    dict  装单条用例（键值查询方便）
    tuple 装固定的枚举值（不可变，防止被误改）
    set   做步骤去重（自动唯一）
"""

from __future__ import annotations

from agent.models import CaseType, Priority, TestCase, TestSuite


# tuple：场景类型固定不变，用 tuple 而不是 list，语义上就表明"别改我"
SCENARIO_TYPES: tuple[str, ...] = ("正常", "异常", "边界")

# tuple of tuple：不可变的模板表，是这个功能点最核心的"业务规则"
LOGIN_TEMPLATES: tuple[tuple[CaseType, Priority, str, tuple[str, ...], str], ...] = (
    (
        "正常",
        "P0",
        "使用已注册账号正常登录",
        ("打开首页", "点击登录入口", "输入正确的邮箱", "输入正确的密码", "点击登录按钮"),
        "登录成功，页面顶部显示 'Logged in as 用户名'",
    ),
    (
        "异常",
        "P1",
        "密码错误时登录失败",
        ("打开首页", "点击登录入口", "输入正确的邮箱", "输入错误的密码", "点击登录按钮"),
        "登录失败，提示 'Your email or password is incorrect!'",
    ),
    (
        "异常",
        "P1",
        "账号不存在时登录失败",
        ("打开首页", "点击登录入口", "输入未注册的邮箱", "输入任意密码", "点击登录按钮"),
        "登录失败，提示邮箱或密码错误",
    ),
    (
        "异常",
        "P1",
        "邮箱为空时登录被拦截",
        ("打开首页", "点击登录入口", "邮箱留空", "输入密码", "点击登录按钮"),
        "无法提交或提示邮箱必填",
    ),
    (
        "异常",
        "P2",
        "密码为空时登录被拦截",
        ("打开首页", "点击登录入口", "输入邮箱", "密码留空", "点击登录按钮"),
        "无法提交或提示密码必填",
    ),
    (
        "边界",
        "P2",
        "密码长度取最小边界值",
        ("打开首页", "点击登录入口", "输入正确邮箱", "输入 1 位密码", "点击登录按钮"),
        "登录失败，不崩溃，给出明确提示",
    ),
    (
        "边界",
        "P2",
        "超长密码不导致系统异常",
        ("打开首页", "点击登录入口", "输入正确邮箱", "输入 500 位超长密码", "点击登录按钮"),
        "登录失败，前端或后端有长度校验，页面不报错",
    ),
    (
        "边界",
        "P3",
        "邮箱含特殊字符时不崩溃",
        ("打开首页", "点击登录入口", "输入含特殊字符的邮箱", "输入密码", "点击登录按钮"),
        "登录失败，提示格式错误，不发生 500 错误",
    ),
)


def dedupe_steps(steps: list[str]) -> list[str]:
    """步骤去重，但保持原来的先后顺序。

    直接用 set 会打乱顺序（set 是无序的），所以这里用 seen 集合辅助：
    遍历时先问 seen 有没有，没有才收进结果。

    这是 Python 里非常常见的一个小套路，值得背下来。
    """
    seen: set[str] = set()
    result: list[str] = []

    for step in steps:
        if step not in seen:
            seen.add(step)
            result.append(step)

    return result


def build_case(
    case_id: str,
    name: str,
    case_type: CaseType,
    priority: Priority,
    steps: list[str],
    expected: str,
) -> TestCase:
    """把零散参数组装成一条标准用例。

    为什么不直接到处写 dict 字面量？
        因为"怎么组装"是规则，规则只写一次。
        以后想加个新字段（比如 precondition），改这一个函数就够了。
    """
    return {
        "id": case_id,
        "name": name,
        "case_type": case_type,
        "priority": priority,
        "steps": dedupe_steps(steps),
        "expected": expected,
    }


def build_login_cases() -> list[TestCase]:
    """生成登录功能的测试用例列表。

    用 f"{i:03d}" 把 1 补成 001 —— f-string 的格式化语法，很实用。
    """
    cases: list[TestCase] = []

    for index, (case_type, priority, name, steps, expected) in enumerate(
        LOGIN_TEMPLATES, start=1
    ):
        cases.append(
            build_case(
                case_id=f"TC_LOGIN_{index:03d}",
                name=name,
                case_type=case_type,
                priority=priority,
                steps=list(steps),
                expected=expected,
            )
        )

    return cases


def build_test_suite(feature: str) -> TestSuite:
    """按功能名分发生成器。

    现在只支持登录，后面加新功能就往 BUILDERS 里加一项。
    这种"表驱动"的写法比一长串 if/elif 好维护。
    """
    builders: dict[str, callable] = {
        "登录": build_login_cases,
        "用户登录": build_login_cases,
    }

    for keyword, builder in builders.items():
        if keyword in feature:
            return {"feature": feature, "cases": builder()}

    raise ValueError(f"暂不支持的功能：{feature}（当前只支持：{list(builders)}）")


def summarize(suite: TestSuite) -> dict[str, int]:
    """统计各类型 / 各优先级的用例数量。

    复习 dict 的 setdefault 用法：key 不存在就先设为 0。
    """
    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}

    for case in suite["cases"]:
        by_type[case["case_type"]] = by_type.get(case["case_type"], 0) + 1
        by_priority[case["priority"]] = by_priority.get(case["priority"], 0) + 1

    return {
        "总数": len(suite["cases"]),
        **{f"类型-{k}": v for k, v in by_type.items()},
        **{f"优先级-{k}": v for k, v in by_priority.items()},
    }


if __name__ == "__main__":
    import json

    suite = build_test_suite("用户登录功能")
    print(json.dumps(suite, ensure_ascii=False, indent=2))
    print("\n统计：", summarize(suite))
