"""
Human-in-the-loop（人工确认）机制（Day 19）

------------------------------------------------------------------
为什么必须有人在环里
------------------------------------------------------------------
让 AI 生成**并执行**代码，听起来很酷，但工程上是很危险的事：

    1. 大模型会"一本正经地写错"——语法没问题，逻辑是错的
    2. 测试代码会真的去操作浏览器、发请求、下单、删数据
    3. 一个跑飞的循环可能反复调用付费 API，几毛几分地烧钱

所以在"生成完"和"执行"之间，必须插一道**人工确认**。

这正是你简历/面试里最值钱的一句话：

    "我没有设计成完全自主执行，而是采用 Human-in-the-loop，
     在代码执行前增加人工确认，降低大模型生成错误代码带来的风险。"

这句话的含金量比"我用了 Multi-Agent"高得多 —— 因为它说明你知道
**能力越大责任越大**，而不是只会堆技术名词。

------------------------------------------------------------------
这个模块就做一件事：在终端上问用户一个问题，拿到明确的答复
------------------------------------------------------------------
"""

from __future__ import annotations

from tools.logger import get_logger

logger = get_logger(__name__)

YES_WORDS = ("y", "yes", "是", "1")
NO_WORDS = ("n", "no", "否", "0")


def ask_yes_no(question: str, default: bool = False) -> bool:
    """问一个是/否问题，返回用户的最终选择。

    参数：
        question  要问的话，例如 "是否执行生成的测试？"
        default   用户直接回车（不输入）时算"是"还是"否"

    default 默认给 False（不同意）——这是刻意的安全设计：
        用户随手敲个回车，不应该等于"同意执行"。
        想执行就得明确输入 y。这叫"安全默认值"。
    """
    suffix = "[Y/n] " if default else "[y/N] "

    while True:
        try:
            answer = input(f"{question} {suffix}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 没有可交互的终端（比如跑在 CI 里），或者用户按了 Ctrl+C
            # 一律按"不同意"处理，绝不擅自执行
            print()
            logger.info("未获得用户输入，按默认答复处理：%s", default)
            return default

        if not answer:
            return default
        if answer in YES_WORDS:
            return True
        if answer in NO_WORDS:
            return False

        print("  输入 y 表示是，n 表示否")


def preview_code(code: str, title: str = "生成的代码") -> None:
    """把代码带行号打印出来，方便人工 review。

    带行号是有实际作用的：
        报错信息里给的就是行号（例如 test_x.py:23），
        带行号打印，你看报错时能立刻对上是哪一行。
    """
    print("=" * 58)
    print(f"  {title}")
    print("=" * 58)

    lines = code.splitlines()
    width = len(str(len(lines)))
    for number, line in enumerate(lines, start=1):
        print(f"  {number:>{width}} | {line}")

    print("=" * 58)


def confirm_execution(code: str, target: str) -> bool:
    """执行前的最后一道关卡：展示代码 + 明确询问。

    这是 Day 19 的主入口。流程是：
        展示代码 -> 告诉用户将要执行什么 -> 问 [y/N] -> 返回结果
    """
    preview_code(code, title=f"待执行：{target}")

    print("\n上面这段代码是 AI 生成的，执行前请确认：")
    print("  - 它操作的是测试环境（automationexercise.com）")
    print("  - 执行过程中会真的打开浏览器")

    return ask_yes_no("\n是否执行这段生成的测试代码？", default=False)
