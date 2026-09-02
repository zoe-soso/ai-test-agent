"""
Agent 循环最小演示（Day 8）

------------------------------------------------------------------
Day 8 只回答一个问题：LLM 和 Agent 到底差在哪？
------------------------------------------------------------------

普通 LLM 调用（一次问答）：

    问题 ──► LLM ──► 回答
                      ▲
                      └── 结束。没有记忆，没有行动，
                          答错了也不知道，更不会自己改。

Agent（带闭环的循环）：

    ┌─────────────────────────────────────────────┐
    │                                             │
    │   目标 ──► 思考 ──► 选工具 ──► 执行          │
    │             ▲                    │          │
    │             │                    ▼          │
    │             └──── 观察结果 ◄──────          │
    │                    │                        │
    │                    └──► 没达成？继续循环     │
    │                         达成了？退出         │
    └─────────────────────────────────────────────┘

差别只有一句话：
    **Agent 能看到自己上一步的结果，并据此决定下一步做什么。**

    这不是什么高深的东西，就是个 while 循环。
    但正因为有这个循环，它才能"失败了重试""缺了就补""做完自己检查"。

------------------------------------------------------------------
这个文件的关键设计：不调 LLM
------------------------------------------------------------------
你可能会问：Agent 的"思考"不是应该由 LLM 来完成吗？

是的。但这里故意用**纯规则**代替 LLM 思考，理由有两个：

1. 今天的目标是看清**循环的骨架**。
   如果把 LLM 调用塞进来，你会被 API 参数、返回解析、异常处理
   这些细节淹没，反而看不清主干。

2. 零成本、完全可复现。
   不用 Key、不花钱、每次运行结果一模一样，方便你反复改着玩。

真实 Agent 里，只需要把 `think()` 换成一次 `client.chat(...)`，
**循环的其余部分一个字都不用改**。这就是为什么先学循环骨架。

------------------------------------------------------------------
跑起来看看
------------------------------------------------------------------
    python -m agent.mini_loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent import testcase_builder
from agent.models import TestCase
from tools.logger import get_logger

logger = get_logger(__name__)

# 本项目约定的三类场景。Agent 的目标就是让这三类都出现。
REQUIRED_TYPES = ("正常", "异常", "边界")


@dataclass
class Step:
    """循环里的一步：思考了什么、做了什么、看到了什么。"""

    index: int
    thought: str
    action: str
    observation: str

    def __str__(self) -> str:
        return (
            f"\n  步骤 {self.index}\n"
            f"    思考 : {self.thought}\n"
            f"    行动 : {self.action}\n"
            f"    观察 : {self.observation}"
        )


@dataclass
class AgentState:
    """Agent 的"记忆"。

    Agent 为什么需要 state？
        因为 LLM 本身没有记忆——你每次调用都要把上下文重新发一遍。
        所以需要你在代码里维护一个 state，每轮循环后更新它，
        下一轮"思考"时把这个 state 一起交给 LLM。

    这就是为什么 Agent 框架（LangGraph 等）都有一个 State 对象：
        它不是框架的发明，而是"LLM 无记忆"这个事实的必然结果。
    """

    goal: str
    feature: str
    cases: list[TestCase] = field(default_factory=list)
    reviewed: bool = False
    issues: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# 工具：Agent 能调用的"手和脚"
# ----------------------------------------------------------------------
# 工具的共同签名：接收 state，修改 state，返回一段"观察结果"文字。
# 这段返回值会喂回给下一轮的思考——这是闭环的关键。

def tool_generate(state: AgentState) -> str:
    """生成一批用例。

    刻意留了个遗憾：只生成"正常"和"异常"，漏掉"边界"。
    这是为了让下一轮的循环有事可做——
    真实场景下模型漏掉某一类场景是家常便饭。
    """
    suite = testcase_builder.build_test_suite(state.feature)
    cases = [c for c in suite["cases"] if c["case_type"] in ("正常", "异常")]
    state.cases = list(cases)
    return f"生成了 {len(cases)} 条用例，但目前只有正常和异常两类"


def tool_supplement(state: AgentState) -> str:
    """补上缺失的场景类型。"""
    present = {c["case_type"] for c in state.cases}
    missing = [t for t in REQUIRED_TYPES if t not in present]
    if not missing:
        return "没有缺失的场景类型，无需补充"

    suite = testcase_builder.build_test_suite(state.feature)
    added = [c for c in suite["cases"] if c["case_type"] in missing]
    state.cases.extend(added)
    return f"补充了 {len(added)} 条 {'/'.join(missing)} 场景用例"


def tool_review(state: AgentState) -> str:
    """质量检查：找出没有步骤或没有预期结果的用例。"""
    issues: list[str] = []
    for case in state.cases:
        if not case["steps"]:
            issues.append(f"{case['id']} 没有操作步骤")
        if not case["expected"]:
            issues.append(f"{case['id']} 没有预期结果")

    state.reviewed = True
    state.issues = issues

    if issues:
        return f"质量检查发现 {len(issues)} 个问题：{'; '.join(issues)}"
    return "质量检查通过，所有用例都有步骤和预期结果"


def tool_fix(state: AgentState) -> str:
    """修掉 review 发现的问题。

    真实 Agent 这里会调 LLM 让它重写；
    这里用规则兜底，保证演示一定能收敛。
    """
    fixed = 0
    for case in state.cases:
        if not case["steps"]:
            case["steps"] = ["打开被测页面", "按用例描述执行操作"]
            fixed += 1
        if not case["expected"]:
            case["expected"] = "页面给出明确提示，无报错"
            fixed += 1

    state.issues = []
    state.reviewed = False  # 改完要重新检查一遍
    return f"修复了 {fixed} 处问题，需要重新检查"


TOOLS: dict[str, Callable[[AgentState], str]] = {
    "generate": tool_generate,
    "supplement": tool_supplement,
    "review": tool_review,
    "fix": tool_fix,
}


# ----------------------------------------------------------------------
# 思考：决定下一步做什么
# ----------------------------------------------------------------------
def think(state: AgentState) -> tuple[str, str]:
    """基于当前 state 决定下一步。返回 (思考内容, 动作名)。

    真实 Agent 里这里是一次 LLM 调用，prompt 里会带上：
        目标 + 当前 state + 可用工具列表 + 每个工具的作用
    然后让 LLM 返回"我要用哪个工具、参数是什么"。

    这里用 if-else 模拟，逻辑和 LLM 输出的结构完全一致。
    """
    if not state.cases:
        return "一条用例都还没有，先生成一批", "generate"

    present = {c["case_type"] for c in state.cases}
    missing = [t for t in REQUIRED_TYPES if t not in present]
    if missing:
        return f"三类场景还缺 {'/'.join(missing)}，补上", "supplement"

    if state.issues:
        return f"上一轮检查发现 {len(state.issues)} 个问题，先修掉", "fix"

    if not state.reviewed:
        return "三类场景都齐了，做一次质量检查", "review"

    return (
        f"目标达成：{len(state.cases)} 条用例，"
        f"三类场景齐全，质量检查通过。可以结束了",
        "finish",
    )


# ----------------------------------------------------------------------
# 循环：Agent 的主干
# ----------------------------------------------------------------------
def run(goal: str, feature: str = "用户登录功能", max_steps: int = 8) -> tuple[AgentState, list[Step]]:
    """跑一个完整的 Agent 循环。

    max_steps 是**必须的**安全阀。
        没有它，Agent 一旦卡在某个"修了又错、错了又修"的循环里，
        就会一直烧你的钱直到破产。真实系统里这叫"防跑飞"。
    """
    state = AgentState(goal=goal, feature=feature)
    steps: list[Step] = []

    for index in range(1, max_steps + 1):
        thought, action = think(state)

        if action == "finish":
            steps.append(Step(index, thought, "finish", "循环结束"))
            break

        tool = TOOLS.get(action)
        if tool is None:
            # 真实 Agent 里这一步非常重要：LLM 可能"幻觉"出一个不存在的工具
            observation = f"工具 {action} 不存在，可用工具：{list(TOOLS)}"
            steps.append(Step(index, thought, action, observation))
            continue

        observation = tool(state)
        steps.append(Step(index, thought, action, observation))
        logger.info("步骤 %d：%s -> %s", index, action, observation)

    return state, steps


def render(state: AgentState, steps: list[Step]) -> str:
    """把整个循环渲染成文本，方便看清楚每一步。"""
    lines = [
        "=" * 62,
        "  Agent 循环演示（思考过程由规则模拟，不调 LLM）",
        "=" * 62,
        f"  目标   : {state.goal}",
        f"  被测   : {state.feature}",
        f"  安全阀 : 最多 8 步",
        "=" * 62,
    ]
    lines.extend(str(step) for step in steps)

    lines.extend(
        [
            "",
            "-" * 62,
            f"  最终结果：{len(state.cases)} 条用例",
        ]
    )
    for case_type in REQUIRED_TYPES:
        count = sum(1 for c in state.cases if c["case_type"] == case_type)
        lines.append(f"    {case_type}场景 : {count} 条")

    if state.issues:
        lines.append(f"  遗留问题：{state.issues}")
    else:
        lines.append("  遗留问题：无")

    lines.append("=" * 62)
    return "\n".join(lines)


if __name__ == "__main__":
    final_state, history = run(
        goal="为登录功能生成覆盖正常、异常、边界三类场景的测试用例",
        feature="用户登录功能",
    )
    print(render(final_state, history))
