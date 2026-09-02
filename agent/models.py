"""
数据模型（Day 2）

知识点：TypedDict
    dict 很灵活，但灵活也意味着容易写错 key。
    TypedDict 让你既保留 dict 的写法，又能声明"这个字典有哪些字段、什么类型"。
    它不是运行时校验（Python 不会帮你拦），但 PyCharm / VS Code 会给你红线提示，
    而且读代码的人一眼就知道数据结构长什么样。

    性价比很高，推荐现在就用起来。

这套结构是**全项目统一的数据契约**：
    - Day 2 的纯 Python 生成器产出它
    - Day 5 的大模型生成器也产出它
    - Day 17 生成 Playwright 代码时消费它
    换了生产者，消费方代码不用改 —— 这就是"先定数据结构"的好处。
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# Literal 表示"只能是这几个值之一"，比 str 更精确，写错了编辑器会提示
CaseType = Literal["正常", "异常", "边界"]
Priority = Literal["P0", "P1", "P2", "P3"]

# Day 10 新增：这条用例用的是哪种测试设计方法。
#
# 为什么加这个字段？因为你的 AI 项目不能只是"我调用了大模型"，
# 得体现测试专业度。让 AI 标注每条用例的设计方法之后，
# 就能多出一个纯测试专业的指标：
#       "AI 生成的用例覆盖了 5 种设计方法中的 4 种"
# 别的 AI 项目做不到，因为他们的作者不懂测试 —— 而这正是你的优势。
#
# "未标注"是兜底值：模型没写就记这个。
# 统计覆盖率时把它单独排除，这样指标不会虚高。
DesignMethod = Literal[
    "等价类划分", "边界值分析", "判定表", "场景法", "异常测试", "未标注"
]

# 真正被认可的设计方法（不含兜底值），评测时用它当覆盖率的分母
DESIGN_METHODS: tuple[str, ...] = (
    "等价类划分", "边界值分析", "判定表", "场景法", "异常测试",
)


class TestCase(TypedDict):
    """一条测试用例。"""

    id: str                 # 用例编号，如 TC_LOGIN_001
    name: str               # 用例名称，如 "正常登录"
    case_type: CaseType     # 正常 / 异常 / 边界
    priority: Priority      # P0 最高，P3 最低
    steps: list[str]        # 操作步骤
    expected: str           # 预期结果

    # Day 10 新增。NotRequired 表示"可以没有这个字段"，
    # 这样 Day 2~9 构造的用例（没有 design_method）依然类型正确，
    # 不用把之前的代码全改一遍。Python 3.11+ 才有这个东西。
    design_method: NotRequired[DesignMethod]


class TestSuite(TypedDict):
    """一个功能点的用例集合。"""

    feature: str            # 功能名，如 "用户登录功能"
    cases: list[TestCase]
