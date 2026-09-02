"""
项目主入口

每天往这里加一个子命令，它就是整个 Agent 的命令行界面。

    python main.py check                 # Day 1  环境自检
    python main.py build                 # Day 3  需求 -> 测试用例 YAML
    python main.py llm "你的问题"         # Day 5  第一次调用大模型
    python main.py prompt-compare        # Day 6  Prompt A/B 对比实验

知识点：argparse 子命令
    比自己解析 sys.argv 好在哪？
        - 自动生成 --help
        - 参数缺失、类型错误时自动报错并给出提示
        - 以后接 CI 时，每个子命令就是一个独立步骤
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import settings
from tools.logger import get_logger

logger = get_logger("main")


def _fix_windows_console_encoding() -> None:
    """Windows 中文控制台编码兜底。

    Windows 控制台默认 GBK，把标准输出强制成 UTF-8，避免 UnicodeEncodeError。
    """
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


# ----------------------------------------------------------------------
# 子命令实现
# ----------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """Day 1：打印环境体检报告。"""
    settings.ensure_dirs()

    print("=" * 58)
    print("  基于 LLM 的 Web 智能测试用例生成与缺陷分析 Agent")
    print("  环境自检报告")
    print("=" * 58)

    for key, value in settings.describe().items():
        print(f"  {key} : {value}")

    print("-" * 58)

    if not settings.LLM_API_KEY:
        print("  [提示] 还没配置 LLM_API_KEY。")
        print("         复制 .env.example 为 .env 并填入 Key，")
        print("         或在 .env 里把 LLM_MOCK 设为 true 先离线演练。")
        return 1

    print("  [OK] 环境就绪。")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Day 3：需求.txt -> Python -> testcases.yaml。"""
    from agent import requirement_reader, testcase_builder
    from tools import file_io

    requirement = requirement_reader.load(args.requirement)
    print(f"[1/4] 读取需求：{requirement.feature}（来源 {requirement.source}）")

    suite = testcase_builder.build_test_suite(requirement.feature)
    print(f"[2/4] 生成用例：{len(suite['cases'])} 条")

    file_io.write_yaml(args.out, suite)
    print(f"[3/4] 写入 YAML：{args.out}")

    # 回读校验：写完再读一遍，确认落盘的东西和内存里的一致。
    # 这一步看着多余，但能第一时间抓出编码、缩进、类型序列化的问题。
    loaded = file_io.read_yaml(args.out)
    assert loaded == suite, "回读校验失败：YAML 内容与原始对象不一致"
    print("[4/4] 回读校验：通过")

    print("\n统计：")
    for key, value in testcase_builder.summarize(suite).items():
        print(f"  {key} : {value}")

    return 0


def cmd_demo_error(args: argparse.Namespace) -> int:
    """Day 4：读取需求 -> 制造异常 -> 记录日志 -> 优雅恢复。

    这个演示想说明一件事：
        AI Agent 调用大模型时，失败是常态而不是意外 ——
        网络抖动、Key 过期、模型返回一堆不是 JSON 的文本……
        能把失败"接住、记下来、告诉用户下一步怎么办"，
        比"让它一次跑通"重要得多。
    """
    import time

    from agent import requirement_reader
    from tools.exceptions import AgentError, RequirementError

    print("=" * 58)
    print("  Day 4 演示：异常处理 + 日志")
    print("=" * 58)

    # ---- 场景 1：文件不存在 ----
    print("\n[场景1] 读取一个不存在的需求文件")
    try:
        requirement_reader.load("data/这个文件不存在.txt")
    except RequirementError as exc:
        logger.error("需求读取失败：%s", exc)
        print(f"  日志已记录 -> {exc}")
        print(f"  给用户看 -> {exc.user_message}")

    # ---- 场景 2：内容为空 ----
    print("\n[场景2] 需求文件是空的")
    try:
        requirement_reader.parse_text("   \n\n  ", source="<demo>")
    except RequirementError as exc:
        logger.error("需求解析失败：%s", exc)
        print(f"  日志已记录 -> {exc}")
        print(f"  给用户看 -> {exc.user_message}")

    # ---- 场景 3：try / except / else / finally 完整走一遍 ----
    print("\n[场景3] try-except-else-finally 的执行顺序")
    start = time.perf_counter()
    try:
        requirement = requirement_reader.load("data/requirement.txt")
    except AgentError as exc:
        # 精确捕获本项目所有业务异常
        logger.error("业务异常：%s", exc)
        print(f"  失败：{exc}")
    else:
        # 只有 try 里没抛异常时才执行。
        # 好处：把"成功才做的事"和 try 块分开，不会被 except 误伤。
        logger.info("需求读取成功：%s", requirement.feature)
        print(f"  成功：读到需求「{requirement.feature}」")
    finally:
        # 无论成功失败都会执行，用来释放资源、记录耗时。
        cost = time.perf_counter() - start
        logger.info("场景3 耗时 %.4f 秒", cost)
        print(f"  finally：耗时 {cost:.4f} 秒（成功失败都会执行）")

    # ---- 场景 4：异常链 ----
    print("\n[场景4] 异常链 raise ... from exc 保留了原始原因")
    try:
        try:
            open("data/绝对不存在.txt", encoding="utf-8").read()
        except OSError as exc:
            raise RequirementError("包装成业务异常", source="demo") from exc
    except AgentError as exc:
        logger.error("异常链：%s | 原始原因：%s", exc, exc.__cause__)
        print(f"  当前异常 -> {type(exc).__name__}: {exc}")
        print(f"  原始原因 -> {type(exc.__cause__).__name__}: {exc.__cause__}")

    print("\n" + "-" * 58)
    print(f"  完整日志见：{settings.LOG_DIR / 'agent.log'}")
    return 0


def cmd_llm(args: argparse.Namespace) -> int:
    """Day 5：第一次调用大模型。"""
    from agent import llm_client
    from tools import file_io
    from tools.exceptions import AgentError

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 5：第一次调用大模型")
    print("=" * 58)
    print(f"  供应商地址 : {client.base_url}")
    print(f"  模型       : {client.model}")
    print(f"  运行模式   : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print(f"  你的输入   : {args.prompt}")
    print("=" * 58)

    try:
        answer = client.chat(args.prompt)
    except AgentError as exc:
        # 精确捕获本项目异常，给用户看"下一步怎么办"，而不是甩一个堆栈
        logger.error("LLM 调用失败：%s", exc)
        print(f"\n[ERROR] {exc}")
        print(f"[提示] {exc.user_message}")
        return 1

    print(answer)
    print("-" * 58)

    # 落盘：第一次真实调用的结果值得留个纪念，也方便对比不同 Prompt 的效果
    record = (
        f"# Day 5 首次调用记录\n\n"
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 模型：{client.model}\n"
        f"- 模式：{'Mock' if client.is_mock else '真实调用'}\n\n"
        f"## 输入\n\n{args.prompt}\n\n"
        f"## 输出\n\n{answer}\n"
    )
    saved = file_io.write_text(settings.OUTPUT_DIR / "llm_first_call.md", record)
    print(f"已保存到：{saved}")
    return 0


def cmd_prompt_compare(args: argparse.Namespace) -> int:
    """Day 6：朴素 Prompt vs 工程化 Prompt 的 A/B 对比实验。"""
    from agent import prompt_lab
    from tools import file_io

    print("=" * 58)
    print("  Day 6：Prompt A/B 对比实验")
    print("=" * 58)
    print(f"  被测功能 : {args.feature}")
    print(f"  对比版本 : {', '.join(args.variants)}")
    print("=" * 58)

    items = prompt_lab.compare(
        feature=args.feature,
        description=args.description,
        variants=args.variants,
    )

    # ---- 分数并排打出来 ----
    print("\n【分数对比】\n")
    metrics = list(items[0].score.keys()) if items else []

    name_width = max(len(i.name) for i in items) if items else 10
    header = f"  {'指标':<12}" + "".join(f"{i.name:>{name_width + 4}}" for i in items)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for metric in metrics:
        row = f"  {metric:<12}"
        for item in items:
            value = item.score.get(metric)
            shown = "OK" if value is True else ("--" if value is False else str(value))
            row += f"{shown:>{name_width + 4}}"
        print(row)

    print("\n【Prompt / 输出 长度】\n")
    for item in items:
        print(f"  {item.name:<32} prompt {item.prompt_chars:>5} 字 -> 输出 {len(item.output):>5} 字")

    # ---- 原始输出 ----
    print("\n" + "=" * 58)
    for item in items:
        print(f"\n----- {item.name} -----")
        print(item.output)

    # ---- 落盘 ----
    report = prompt_lab.render_report(items, args.feature)
    saved = file_io.write_text(settings.OUTPUT_DIR / "prompt_compare.md", report)
    print("\n" + "=" * 58)
    print(f"对比报告已保存到：{saved}")
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    """Day 9：需求 -> 生成 -> 覆盖自检 -> 定向补充 -> YAML。"""
    from agent import llm_client
    from agent.testcase_generator import TestCaseGenerator

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 9：测试用例生成模块")
    print("=" * 58)
    print(f"  功能     : {args.feature}")
    print(f"  模板     : {args.template}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 58)

    generator = TestCaseGenerator(
        client=client, template=args.template, max_repairs=args.max_repairs
    )
    suite, report = generator.generate(args.feature, args.description)
    saved = generator.save(suite)

    print("\n【场景覆盖自检】")
    for key, value in report.to_dict().items():
        print(f"  {key} : {value}")

    if report.missing:
        print(f"\n  [!] 仍缺少「{'、'.join(report.missing)}」场景，"
              f"需要人工补或改 prompt。")
    else:
        print("\n  [OK] 正常 / 异常 / 边界 三类场景均已覆盖。")

    if report.supplemented:
        print(f"  [i] 通过定向补充拿到了 {report.supplemented} 条（而非全量重生成）。")

    print(f"\n【用例 {len(suite['cases'])} 条】")
    for case in suite["cases"]:
        print(f"  [{case['case_type']}/{case['priority']}] {case['id']} {case['name']}")
        print(f"      预期 -> {case['expected']}")

    print(f"\n已写入 YAML：{saved}")
    print(f"本次消耗：{client.usage}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Day 7：跑一遍评测集，算结构可用率 / 场景覆盖率 / 成本。"""
    from eval import run_eval

    return run_eval.main(template=args.template, max_repairs=args.max_repairs)


# ----------------------------------------------------------------------
# 命令行组装
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from agent import prompt_lab, structured

    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="基于 LLM 的 Web 智能测试用例生成与缺陷分析 Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="环境自检").set_defaults(func=cmd_check)

    sub.add_parser(
        "demo-error",
        help="Day 4 演示：异常处理与日志",
    ).set_defaults(func=cmd_demo_error)

    p_build = sub.add_parser("build", help="需求 -> 测试用例 YAML")
    p_build.add_argument(
        "--requirement",
        default=str(root / "data" / "requirement.txt"),
        help="需求文件路径",
    )
    p_build.add_argument(
        "--out",
        default=str(root / "outputs" / "testcases.yaml"),
        help="输出的 YAML 路径",
    )
    p_build.set_defaults(func=cmd_build)

    p_llm = sub.add_parser("llm", help="直接问大模型一句话")
    p_llm.add_argument("prompt", help="要问的内容，用引号包起来")
    p_llm.set_defaults(func=cmd_llm)

    p_cmp = sub.add_parser(
        "prompt-compare",
        help="Day 6 演示：朴素 Prompt vs 工程化 Prompt 对比",
    )
    p_cmp.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_cmp.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_cmp.add_argument(
        "--variants",
        nargs="+",
        default=list(prompt_lab.DEFAULT_VARIANTS),
        help="要对比的 prompt 模板名（不含 .txt）",
    )
    p_cmp.set_defaults(func=cmd_prompt_compare)

    p_gen = sub.add_parser(
        "gen", help="Day 7 演示：需求 -> LLM -> JSON -> 校验 -> YAML"
    )
    p_gen.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_gen.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_gen.add_argument(
        "--template", default=structured.DEFAULT_TEMPLATE, help="prompt 模板名"
    )
    p_gen.add_argument("--max-repairs", type=int, default=1, help="最多自修正几次")
    p_gen.set_defaults(func=cmd_gen)

    p_eval = sub.add_parser("eval", help="Day 7 演示：跑评测集并记入 history.csv")
    p_eval.add_argument(
        "--template", default=structured.DEFAULT_TEMPLATE, help="prompt 模板名"
    )
    p_eval.add_argument("--max-repairs", type=int, default=1, help="最多自修正几次")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> int:
    _fix_windows_console_encoding()
    settings.ensure_dirs()

    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
