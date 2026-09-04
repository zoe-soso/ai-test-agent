"""
项目主入口

每天往这里加一个子命令，它就是整个 Agent 的命令行界面。

    python main.py check                 # Day 1  环境自检
    python main.py build                 # Day 3  需求 -> 测试用例 YAML
    python main.py llm "你的问题"         # Day 5  第一次调用大模型
    python main.py prompt-compare        # Day 6  Prompt A/B 对比实验
    python main.py review --fix          # Day 11 用例质量评审（并自动修改）

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
from tools.exceptions import AgentError
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


def _load_cases_for_review(
    args: argparse.Namespace, client: object
) -> tuple[list, str]:
    """取待评审的用例（返回 用例列表 + 来源说明）。

    两种来源：
        1. --input 指定了 YAML —— 评审已有的用例（实际最常用的姿势）
        2. 没指定 —— 走一遍生成链路现造一批（一条命令演示全流程时方便）
    """
    from agent import structured, validator
    from tools import file_io

    if args.input:
        data = file_io.read_yaml(args.input)
        raws = data.get("cases", []) if isinstance(data, dict) else data
        passed, _, failed = validator.validate_cases(raws, feature_key="AUTO")
        note = f"文件 {args.input}"
        if failed:
            note += f"（{len(failed)} 条不合格已剔除）"
        return passed, note

    result = structured.generate(
        args.feature, args.description,
        template=args.template, client=client,  # type: ignore[arg-type]
    )
    return result.cases, f"现场生成（{result.attempts} 次调用）"


def cmd_review(args: argparse.Namespace) -> int:
    """Day 11：生成 -> Review -> 修改 的质量闭环。"""
    from agent import llm_client, reviewer
    from tools import file_io

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 11：测试用例质量检查（Review）")
    print("=" * 58)
    print(f"  功能     : {args.feature}")
    print(f"  LLM 评审 : {'开启' if not args.rule_only else '关闭（仅规则层，零成本）'}")
    print(f"  修改闭环 : {'开启' if args.fix else '关闭（只评审，不改用例）'}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 58)

    # ---- 1. 取待评审用例 ----
    try:
        cases, source = _load_cases_for_review(args, client)
    except Exception as exc:  # noqa: BLE001
        logger.error("读取待评审用例失败：%s", exc)
        print(f"\n[ERROR] 读取用例失败：{exc}")
        return 1

    if not cases:
        print("\n[ERROR] 没有可评审的用例。")
        return 1

    print(f"\n[1/3] 待评审用例 {len(cases)} 条（来源：{source}）")

    # ---- 2. 评审（可选：评审后自动修改）----
    use_llm = not args.rule_only
    if args.fix:
        final, report, rounds = reviewer.review_and_revise(
            args.feature, cases, client=client,
            max_rounds=args.max_rounds, use_llm=use_llm,
        )
    else:
        report = reviewer.review(args.feature, cases, client=client, use_llm=use_llm)
        final, rounds = cases, 0

    print(f"[2/3] 评审完成：{report.describe()}")
    print("\n【评审摘要】")
    for key, value in report.summary().items():
        print(f"  {key} : {value}")

    # ---- 3. 问题明细 ----
    if report.issues:
        print("\n【问题清单】")
        for issue in report.sorted_issues():
            print(f"  {issue}")
    else:
        print("\n【问题清单】无 —— 用例质量合格")

    if report.overall:
        print(f"\n【LLM 总评】{report.overall}")
    if report.missing_scenarios:
        print(f"\n【遗漏场景】{'、'.join(report.missing_scenarios)}")

    # ---- 4. 落盘 ----
    print("\n[3/3] 结果处理")
    if args.fix:
        suite = {"feature": args.feature, "cases": final}
        out = Path(args.out)
        file_io.write_yaml(out, suite)
        print(f"  修改轮数 : {rounds}")
        print(f"  用例变化 : {len(cases)} 条 -> {len(final)} 条")
        print(f"  已写入   : {out}")
    else:
        print("  未开启 --fix，用例未改动（加 --fix 可让 LLM 按意见修改）。")

    print(f"\n本次消耗：{client.usage}")
    return 0


def _preview_fields(fields: dict) -> str:
    """把一组数据压成一行预览。

    超长值必须截断：500 字符的密码会把整个控制台刷屏，
    而且那种情况下你真正需要知道的只是"它有多长"。
    """
    parts = []
    for key, value in fields.items():
        text = str(value)
        if len(text) > 40:
            text = f"{text[:37]}...（共 {len(str(value))} 字符）"
        parts.append(f'{key}="{text}"')
    return "  ".join(parts)


def cmd_gen_data(args: argparse.Namespace) -> int:
    """Day 12：生成测试数据。"""
    from agent import llm_client, validator
    from agent.testdata_generator import TestDataGenerator
    from tools import file_io

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 12：测试数据生成")
    print("=" * 58)
    print(f"  功能     : {args.feature}")
    print(f"  参数     : {'、'.join(args.params) if args.params else '（由模型判断）'}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 58)

    # 可选：加载用例，把数据挂到用例上
    cases = None
    if args.link:
        try:
            payload = file_io.read_yaml(args.link)
            raws = payload.get("cases", []) if isinstance(payload, dict) else payload
            cases, _, _ = validator.validate_cases(raws, feature_key="AUTO")
            print(f"\n[关联] 已加载 {len(cases)} 条用例，数据将挂到对应用例上")
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载关联用例失败，将跳过关联：%s", exc)

    generator = TestDataGenerator(client=client, max_repairs=args.max_repairs)
    suite, report = generator.generate(
        args.feature, args.description,
        params=args.params or None, cases=cases,
    )
    saved = generator.save(suite, args.out)

    print("\n【生成摘要】")
    for key, value in report.to_dict().items():
        print(f"  {key} : {value}")

    # 数据质量检查（纯规则，不花钱）
    problems = generator.quality_check(suite["data"])
    if problems:
        print(f"\n【数据质量检查】发现 {len(problems)} 处问题：")
        for problem in problems:
            print(f"  [!] {problem}")
    else:
        print("\n【数据质量检查】未发现占位符数据，超长数据长度也达标")

    print(f"\n【测试数据 {len(suite['data'])} 组】")
    for item in suite["data"]:
        link = item.get("link_case")
        suffix = f"  -> {link}" if link else ""
        print(f"  [{item['data_type']}] {item['id']} {item['name']}{suffix}")
        print(f"      {_preview_fields(item['fields'])}")

    print(f"\n已写入 YAML：{saved}")
    print(f"本次消耗：{client.usage}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Day 7：跑一遍评测集，算结构可用率 / 场景覆盖率 / 成本。"""
    from eval import run_eval

    return run_eval.main(template=args.template, max_repairs=args.max_repairs)


def cmd_agent(args: argparse.Namespace) -> int:
    """Day 14：第一个可演示的 Agent MVP（生成 → Review → 落盘）。"""
    from agent import llm_client
    from agent.planner import TestCaseAgent

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 14：测试用例生成 Agent（MVP）")
    print("=" * 58)
    print(f"  需求     : {args.feature}")
    print(f"  自动修改 : {'开启' if not args.no_fix else '关闭'}")
    print(f"  LLM 评审 : {'开启' if not args.rule_only else '关闭（仅规则层）'}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 58)

    agent = TestCaseAgent(
        client=client,
        template=args.template,
        auto_fix=not args.no_fix,
        use_llm=not args.rule_only,
        max_repairs=args.max_repairs,
    )

    try:
        result = agent.run(args.feature, args.description,
                           cases_path=args.out, review_path=args.review_out)
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent 运行失败：%s", exc, exc_info=True)
        print(f"\n[ERROR] {exc}")
        return 1

    print("\n【运行结果】")
    for key, value in result.summary().items():
        print(f"  {key} : {value}")

    if result.review.issues:
        print("\n【评审问题】")
        for issue in result.review.sorted_issues():
            print(f"  {issue}")
    else:
        print("\n【评审问题】无 —— 用例质量合格")

    if result.review.overall:
        print(f"\n【LLM 总评】{result.review.overall}")

    print(f"\n已写入：\n  - 用例  : {result.cases_path}\n  - 评审  : {result.review_path}")
    print(f"本次消耗：{client.usage}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """Day 13：Tool Calling 演示 —— 让模型自己决定调用哪个工具。

    这条命令不预设流程，而是把一个自由指令交给 Agent：
    模型读完工具清单后，自己决定调不调、先调哪个。
    这是 Tool Calling 和"写死流程"最大的区别 —— 控制权在模型手里。

    示例：
        python main.py tools "帮我把用户登录功能的测试用例生成并保存"
    """
    from agent import llm_client
    from agent import agent_tools
    from agent.tool_calling import run_tool_loop

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 13：Tool Calling 演示（工具循环）")
    print("=" * 58)
    print(f"  可用工具 : {', '.join(agent_tools.registry.names)}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print(f"  指令     : {args.instruction}")
    print("=" * 58)

    messages = [
        {"role": "system", "content": agent_tools.build_system_prompt()},
        {"role": "user", "content": args.instruction},
    ]

    result = run_tool_loop(
        client, messages, agent_tools.registry,
        max_iterations=args.max_iterations,
    )

    print("\n【工具调用轨迹】")
    if result.steps:
        for step in result.steps:
            print(step)
    else:
        print("  （模型没有调用任何工具，直接回答了）")

    print(f"\n【循环轮数】{result.iterations} ｜ 结束原因：{result.stopped_reason}")
    print(f"\n【模型最终答复】\n{result.answer}")
    return 0


def _load_cases_for_code(args: argparse.Namespace, client: object) -> tuple[list, str]:
    """取待转代码的用例（返回 用例列表 + 来源说明）。

    两种来源：
        1. --input 指定了 YAML —— 用已有的用例（最常用，也最省钱）
        2. 没指定 —— 先生成一批用例，再转成代码（一条命令演示全链路）
    """
    from agent import validator
    from agent.testcase_generator import TestCaseGenerator
    from tools import file_io

    if args.input:
        payload = file_io.read_yaml(args.input)
        raws = payload.get("cases", []) if isinstance(payload, dict) else payload
        passed, _, _ = validator.validate_cases(raws, feature_key="AUTO")
        return passed, f"文件 {args.input}"

    generator = TestCaseGenerator(client=client)  # type: ignore[arg-type]
    suite, _ = generator.generate(args.feature, args.description)
    return suite["cases"], "现场生成"


def cmd_code(args: argparse.Namespace) -> int:
    """Day 16~20：测试用例 -> Playwright 代码 -> 人工确认 -> 执行 pytest。"""
    from agent import llm_client
    from agent.code_generator import CodeGenerator
    from tools import test_runner
    from tools.human import ask_yes_no, confirm_execution

    client = llm_client.get_client()

    print("=" * 58)
    print("  Day 16~20：测试用例 → Playwright 代码 → 执行")
    print("=" * 58)
    print(f"  功能     : {args.feature}")
    print(f"  生成条数 : {args.limit}")
    print(f"  执行测试 : {'是（会先让你确认）' if args.run else '否'}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 58)

    # ---- 1. 取用例 ----
    try:
        cases, source = _load_cases_for_code(args, client)
    except Exception as exc:  # noqa: BLE001
        logger.error("读取用例失败：%s", exc)
        print(f"\n[ERROR] 读取用例失败：{exc}")
        return 1

    if not cases:
        print("\n[ERROR] 没有可用的用例，无法生成代码。")
        return 1

    print(f"\n[1/4] 待转换用例 {len(cases)} 条（来源：{source}）")

    # ---- 2. 生成代码 ----
    generator = CodeGenerator(client=client, max_repairs=args.max_repairs)
    results = generator.generate_many(args.feature, cases, limit=args.limit)
    print("[2/4] 代码生成完成")

    # ---- 3. 代码检查 + 保存 ----
    saved: list[str] = []
    for item in results:
        print(f"\n  {item.describe()}")
        for issue in item.issues:
            print(f"      [!] {issue}")
        if not item.code:
            continue

        if args.yes or ask_yes_no(f"      是否保存 {item.filename}？", default=True):
            path = generator.save(item)
            saved.append(str(path))
            print(f"      已保存 -> {path}")

    if not saved:
        print("\n[3/4] 没有任何代码被保存，结束。")
        return 1
    print(f"\n[3/4] 共保存 {len(saved)} 个文件")

    # ---- 4. 人工确认 + 执行（Day 19 + Day 20）----
    if not args.run:
        print("\n[4/4] 未加 --run，不执行（这是刻意的：执行前必须人确认）。")
        print("      想执行请加 --run，会先展示代码让你确认。")
        return 0

    target = args.target or saved[0]
    code_text = Path(target).read_text(encoding="utf-8")

    if args.yes:
        print(f"\n[4/4] --yes 已跳过人工确认，直接执行：{target}")
    elif not confirm_execution(code_text, target):
        print("\n[4/4] 你选择了不执行。代码已保存在原处，随时可以手动跑。")
        return 0

    print(f"\n正在执行：{target}（浏览器 {args.browser}）")
    result = test_runner.run_pytest(target, browser=args.browser, timeout=args.timeout)

    print("\n【执行结果】")
    for key, value in result.to_dict().items():
        print(f"  {key} : {value}")

    if not result.success and result.stdout:
        print("\n【pytest 输出（最后 25 行）】")
        for line in result.stdout.strip().splitlines()[-25:]:
            print(f"  {line}")

    return 0 if result.success else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Day 27：把整条流水线跑一遍（项目核心 Demo）。"""
    from agent import llm_client
    from agent.pipeline import run_pipeline

    client = llm_client.get_client()

    print("=" * 60)
    print("  Day 27 完整流水线：需求 → 用例 → 代码 → 执行 → AI 分析")
    print("=" * 60)
    print(f"  功能     : {args.feature}")
    print(f"  生成条数 : {args.limit}")
    print(f"  Allure   : {'是' if args.allure else '否'}")
    print(f"  人工确认 : {'--auto 已跳过' if args.auto else '执行前会让你确认（Human-in-the-loop）'}")
    print(f"  运行模式 : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 60)

    run_pipeline(
        client,
        feature=args.feature,
        description=args.description,
        limit=args.limit,
        auto=args.auto,
        browser=args.browser,
        timeout=args.timeout,
        allure=args.allure,
        max_repairs=args.max_repairs,
    )
    return 0



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

    p_review = sub.add_parser(
        "review", help="Day 11：评审测试用例质量（加 --fix 让 LLM 按意见修改）"
    )
    p_review.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_review.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_review.add_argument(
        "--input", default="",
        help="待评审的用例 YAML 文件；不填则现场生成一批",
    )
    p_review.add_argument(
        "--template", default=structured.DEFAULT_TEMPLATE,
        help="现场生成用例时用的 prompt 模板名",
    )
    p_review.add_argument(
        "--fix", action="store_true", help="评审后让 LLM 按意见修改用例",
    )
    p_review.add_argument("--max-rounds", type=int, default=1, help="最多修改几轮")
    p_review.add_argument(
        "--rule-only", action="store_true",
        help="评审阶段不调用 LLM，只跑规则层（配合 --input 时为真正的零成本）",
    )
    p_review.add_argument(
        "--out",
        default=str(root / "outputs" / "testcases_reviewed.yaml"),
        help="--fix 时，修改后用例的输出路径",
    )
    p_review.set_defaults(func=cmd_review)

    p_data = sub.add_parser(
        "gen-data", help="Day 12：生成测试数据（数据驱动测试的基础）"
    )
    p_data.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_data.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_data.add_argument(
        "--params", nargs="*", default=[],
        help="参数名清单，如 --params username password；不填则由模型判断",
    )
    p_data.add_argument(
        "--link", default="",
        help="用例 YAML 路径；填了就把数据关联到对应用例上",
    )
    p_data.add_argument("--max-repairs", type=int, default=1, help="最多自修正几次")
    p_data.add_argument(
        "--out",
        default=str(root / "outputs" / "testdata.yaml"),
        help="输出的 YAML 路径",
    )
    p_data.set_defaults(func=cmd_gen_data)

    p_eval = sub.add_parser("eval", help="Day 7 演示：跑评测集并记入 history.csv")
    p_eval.add_argument(
        "--template", default=structured.DEFAULT_TEMPLATE, help="prompt 模板名"
    )
    p_eval.add_argument("--max-repairs", type=int, default=1, help="最多自修正几次")
    p_eval.set_defaults(func=cmd_eval)

    p_agent = sub.add_parser(
        "agent", help="Day 14：测试用例生成 Agent MVP（生成 → Review → 落盘）"
    )
    p_agent.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_agent.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_agent.add_argument(
        "--template", default=structured.DEFAULT_TEMPLATE, help="生成用例用的 prompt 模板名"
    )
    p_agent.add_argument(
        "--no-fix", action="store_true",
        help="生成后不自动按评审意见修改（只评审、不改）",
    )
    p_agent.add_argument(
        "--rule-only", action="store_true",
        help="评审阶段不调用 LLM，只跑规则层（零成本）",
    )
    p_agent.add_argument("--max-repairs", type=int, default=1, help="生成/修改最多自修正几次")
    p_agent.add_argument(
        "--out",
        default=str(root / "outputs" / "testcases_ai.yaml"),
        help="生成的用例输出路径",
    )
    p_agent.add_argument(
        "--review-out",
        default=str(root / "outputs" / "review_report.yaml"),
        help="评审报告输出路径",
    )
    p_agent.set_defaults(func=cmd_agent)

    p_tools = sub.add_parser(
        "tools", help="Day 13：Tool Calling 演示（模型自主决定调哪个工具）"
    )
    p_tools.add_argument(
        "instruction",
        help="交给 Agent 的自然语言指令，例如 \"帮我把登录功能的测试用例生成并保存\"",
    )
    p_tools.add_argument("--max-iterations", type=int, default=5, help="工具循环最多几轮")
    p_tools.set_defaults(func=cmd_tools)

    p_code = sub.add_parser(
        "code", help="Day 16~20：测试用例 → Playwright 代码 → 人工确认 → 执行"
    )
    p_code.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_code.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_code.add_argument(
        "--input", default="",
        help="用例 YAML 路径；不填则现场生成一批用例再转代码",
    )
    p_code.add_argument("--limit", type=int, default=1, help="转换前几条用例（默认 1 条，省钱）")
    p_code.add_argument("--max-repairs", type=int, default=1, help="代码检查不合格时最多改几轮")
    p_code.add_argument(
        "--run", action="store_true",
        help="生成后执行测试（执行前会展示代码让你确认）",
    )
    p_code.add_argument(
        "--target", default="",
        help="要执行的文件；默认执行本次保存的第一个文件",
    )
    p_code.add_argument("--browser", default="chromium", help="只用哪种浏览器跑，默认 chromium")
    p_code.add_argument("--timeout", type=int, default=300, help="执行超时秒数")
    p_code.add_argument(
        "--yes", action="store_true",
        help="所有确认自动选是（CI 用；本地演示时不建议，会跳过人工确认）",
    )
    p_code.set_defaults(func=cmd_code)

    p_run = sub.add_parser(
        "run", help="Day 27 核心 Demo：需求→用例→代码→执行→AI 缺陷分析 全链路"
    )
    p_run.add_argument("--feature", default="用户登录功能", help="被测功能名")
    p_run.add_argument("--description", default="（无额外描述）", help="需求描述")
    p_run.add_argument("--limit", type=int, default=1, help="最多生成几条用例对应的代码（默认 1 条，省钱）")
    p_run.add_argument("--max-repairs", type=int, default=1, help="生成/修改/代码检查最多自修正几次")
    p_run.add_argument("--browser", default="chromium", help="只用哪种浏览器跑，默认 chromium")
    p_run.add_argument("--timeout", type=int, default=300, help="执行超时秒数")
    p_run.add_argument(
        "--allure", action="store_true",
        help="执行时产出 Allure 原始结果（Day 21）；报告生成需另装 allure CLI",
    )
    p_run.add_argument(
        "--auto", action="store_true",
        help="跳过人工确认直接执行（CI/演示用；平时不建议，违反 Human-in-the-loop）",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    """统一入口。

    这里做**顶层异常兜底**，是刻意的设计：
        每个子命令内部只管自己的正常流程，
        "出错了要给用户看什么"这件事在这一处统一处理。
    否则每个命令都要写一遍 try/except，而且总有一天会漏一个 ——
    漏掉的那个就是用户看到一整屏红色堆栈的时候。
    """
    _fix_windows_console_encoding()
    settings.ensure_dirs()

    args = build_parser().parse_args()

    try:
        return args.func(args)
    except AgentError as exc:
        # 本项目自己的业务异常：都带 user_message，告诉用户下一步怎么办
        logger.error("命令执行失败：%s", exc, exc_info=True)
        print(f"\n[ERROR] {exc}")
        print(f"[提示] {exc.user_message}")
        return 1
    except KeyboardInterrupt:
        print("\n已取消。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
