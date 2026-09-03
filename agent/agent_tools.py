"""
Agent 的具体工具实现（Day 13）

`tool_calling.py` 提供的是**机制**（Schema 怎么生成、循环怎么跑）；
这个文件提供的是**业务能力**（Agent 到底能干哪些事）。

两者分开，是因为机制是通用的、几乎不会变；
而工具会随着项目推进不断增删 —— 分开改起来不互相干扰。

------------------------------------------------------------------
本文件最重要的一个设计决定：工作区（_WORKSPACE）
------------------------------------------------------------------
你可能会想：工具之间怎么传递数据？

    方案 A（我一开始想的）：
        generate_testcase() 返回完整 JSON 字符串给模型
        → 模型把这段 JSON 原样传给 save_yaml(cases_json=...)
        → save_yaml 再解析、落盘

    这个方案是**错的**，有三个致命问题：
        1. 贵。同样一份数据要付两次 token（模型读一次、模型写一次）
        2. 不可靠。模型"复述"长 JSON 时会漏字段、改缩进、
           甚至自作主张改内容 —— 你拿到手的已经不是生成的那份了
        3. 容易被截断。2000 字的 JSON 直接吃掉大半个输出预算

    方案 B（本项目采用）：工具之间用**工作区**传递数据
        generate_testcase() 生成完，自己存进工作区，只把摘要返回给模型
        save_yaml() 直接从工作区取，落盘

        模型看到的只有"生成了 8 条用例，存为 cases"这样的摘要，
        真正的用例数据**一次都没有经过模型**。

    这不是偷懒，这是 Agent 工程里的通用做法：
    **让模型做决策，让代码搬数据。**
    模型的 token 很贵，而且它会"记错"；Python 变量既免费又精确。

------------------------------------------------------------------
关于 run_pytest 的安全性
------------------------------------------------------------------
让 AI 去执行 shell 命令是很危险的事。本项目做了三层限制：

    1. 只跑 pytest，命令是固定的，模型传不进任意命令
    2. 路径必须落在项目目录内（挡住 ../../ 这类路径穿越）
    3. 有超时（默认 300 秒），不会挂死

三层加起来，模型能造成的破坏被限制在"跑一次测试"的范围内。
Day 19 的 Human-in-the-loop 会再加一道：危险操作要人工点确认。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent import llm_client
from agent.tool_calling import ToolRegistry
from config import settings
from tools import file_io
from tools.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 工作区：工具之间交换数据的地方
# ----------------------------------------------------------------------
# 用模块级 dict 而不是搞一个 Context 类传来传去，
# 是为了让每个工具函数的签名保持"只跟模型要的参数有关"——
# 模型看到的就是工具真正需要的参数，不会被额外的上下文参数干扰。
_WORKSPACE: dict[str, Any] = {}


def reset_workspace() -> None:
    """清空工作区（主要给单元测试用，保证用例之间互不污染）。"""
    _WORKSPACE.clear()


def workspace_keys() -> list[str]:
    """当前工作区里有哪些数据。"""
    return list(_WORKSPACE)


# ----------------------------------------------------------------------
# 路径安全
# ----------------------------------------------------------------------
def _safe_path(raw: str) -> Path:
    """把用户/模型给的路径限制在项目目录内。

    为什么必须有这个函数？
        模型是被 prompt 驱动的，而 prompt 里有用户输入的自然语言。
        如果不做限制，一句"读一下 C:/Windows/System32/config/sam"
        就可能真的被执行 —— 这就是典型的**路径穿越**风险。

    做法很朴素：
        1. resolve() 把 ../ ./ 这些全部展开成绝对路径
        2. 判断它是否还在 PROJECT_ROOT 里面
        3. 不在就直接拒绝
    """
    path = (settings.PROJECT_ROOT / raw).resolve()
    root = settings.PROJECT_ROOT.resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"路径越界：{raw}（解析后为 {path}）不在项目目录 {root} 内，已拒绝访问"
        )
    return path


# 相对 PROJECT_ROOT 显示，避免把整台机器的绝对路径暴露给模型
def _display(path: Path) -> str:
    try:
        return str(path.relative_to(settings.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------
# 工具注册
# ----------------------------------------------------------------------
registry = ToolRegistry()


@registry.tool
def generate_testcase(feature: str, description: str = "（无额外描述）") -> str:
    """为指定功能生成测试用例，结果存入工作区。

    这是本 Agent 的核心能力。生成的用例会存放在工作区里，
    之后可以用 save_yaml 落盘，或者继续让 Agent 做别的处理。
    本工具只返回摘要，不返回完整用例内容。

    Args:
        feature: 被测功能名，例如"用户登录功能""购物车结算"
        description: 功能的补充说明，例如"邮箱+密码登录，支持记住我"
    """
    from agent.testcase_generator import TestCaseGenerator

    client = llm_client.get_client()
    generator = TestCaseGenerator(client=client)
    suite, report = generator.generate(feature, description)

    cases = suite["cases"]
    _WORKSPACE["cases"] = cases
    _WORKSPACE["feature"] = feature

    lines = [
        f"已生成 {len(cases)} 条用例，功能：{feature}（已存入工作区，键名 cases）",
        f"场景覆盖：正常 {report.counts.get('正常', 0)} / "
        f"异常 {report.counts.get('异常', 0)} / "
        f"边界 {report.counts.get('边界', 0)} 条",
        "用例清单：",
    ]
    for case in cases:
        method = case.get("design_method") or "-"
        lines.append(
            f"  [{case['case_type']}/{case['priority']}/{method}] "
            f"{case['id']} {case['name']}"
        )

    if report.missing:
        lines.append(f"[警告] 仍缺少场景：{'、'.join(report.missing)}")
    if report.supplemented:
        lines.append(f"[说明] 其中 {report.supplemented} 条是定向补充得来的")

    return "\n".join(lines)


@registry.tool
def save_yaml(filename: str = "testcases.yaml") -> str:
    """把工作区里的测试用例保存成 YAML 文件。

    必须先调用 generate_testcase 生成用例，否则工作区是空的。

    Args:
        filename: 输出文件名，会自动保存到 outputs 目录下，例如 testcases.yaml
    """
    cases = _WORKSPACE.get("cases")
    if not cases:
        return (
            "[错误] 工作区里没有用例。"
            "请先调用 generate_testcase 生成用例，再调用本工具保存。"
        )

    # 只取文件名，防止模型传进来的路径带 ../ 写到项目外面去。
    # 这一行看着不起眼，但没有它，模型就有能力往任意位置写文件。
    name = Path(filename).name or "testcases.yaml"
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"

    target = settings.OUTPUT_DIR / name
    file_io.write_yaml(target, {"feature": _WORKSPACE.get("feature", ""), "cases": cases})
    logger.info("工具落盘：%s（%d 条用例）", target, len(cases))

    return f"已保存 {len(cases)} 条用例到 outputs/{name}（完整路径：{target}）"


@registry.tool
def read_file(path: str) -> str:
    """读取项目内的一个文本文件内容。

    只能读取项目目录内的文件，路径中不允许包含 .. 跳出项目目录。

    Args:
        path: 相对项目根目录的路径，例如 data/requirement.txt 或 outputs/testcases.yaml
    """
    try:
        target = _safe_path(path)
    except ValueError as exc:
        return f"[错误] {exc}"

    if not target.exists():
        return f"[错误] 文件不存在：{path}（解析为 {_display(target)}）"
    if not target.is_file():
        return f"[错误] 这不是一个文件：{path}"

    # 内容上限 8000 字符。
    # 把整个大文件塞给模型既烧钱又占上下文，而绝大多数情况下
    # 它需要的只是开头部分。截断时明确告诉模型"被截断了"，
    # 否则它会以为这就是文件的全部内容，然后得出错误结论。
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"[错误] 读取失败：{type(exc).__name__}: {exc}"

    if len(content) > 8000:
        content = content[:8000] + f"\n\n...（已截断，文件共 {len(content)} 字符）"

    return f"文件 {_display(target)}（{len(content)} 字符）：\n\n{content}"


@registry.tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """列出项目内某个目录下的文件。

    用来让 Agent 了解项目结构，例如想知道有哪些测试用例文件、有哪些 prompt 模板。

    Args:
        directory: 相对项目根目录的目录路径，例如 tests 或 prompts，默认是项目根目录
        pattern: 文件名匹配模式，例如 *.py、*.yaml、*.txt
    """
    try:
        target = _safe_path(directory)
    except ValueError as exc:
        return f"[错误] {exc}"

    if not target.is_dir():
        return f"[错误] 不是一个目录：{directory}"

    files = sorted(p for p in target.glob(pattern) if p.is_file())
    if not files:
        return f"目录 {_display(target)} 下没有匹配 {pattern} 的文件"

    lines = [f"目录 {_display(target)} 下有 {len(files)} 个匹配 {pattern} 的文件："]
    for path in files[:50]:
        lines.append(f"  {_display(path)}  （{path.stat().st_size} 字节）")
    if len(files) > 50:
        lines.append(f"  ...（还有 {len(files) - 50} 个未列出）")
    return "\n".join(lines)


@registry.tool
def run_pytest(target: str = "tests", timeout: int = 300) -> str:
    """执行 pytest 并只返回结果摘要。

    注意：本工具返回的是**精简摘要**，不是完整的 pytest 输出。
    完整输出动辄几千行，会直接撑爆上下文；而模型真正需要的
    其实只有"过了几条、挂了哪几条、报什么错"。

    Args:
        target: 要执行的测试路径，相对项目根目录，例如 tests、tests/test_day01_05.py
        timeout: 超时秒数，默认 300
    """
    try:
        path = _safe_path(target)
    except ValueError as exc:
        return f"[错误] {exc}"

    if not path.exists():
        return f"[错误] 路径不存在：{target}"

    # 固定命令：模型只能通过 target 参数影响范围，塞不进任何 shell 语法。
    # 这是"让 AI 执行命令"时最要紧的一条纪律 —— 命令模板写死，只留参数口子。
    command = [sys.executable, "-m", "pytest", str(path), "-q", "--no-header"]
    logger.info("工具执行：%s", " ".join(command))

    try:
        completed = subprocess.run(
            command,
            cwd=str(settings.PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(max(timeout, 10), 600),
        )
    except subprocess.TimeoutExpired:
        return f"[错误] 执行超时（>{timeout} 秒）。可改用更小的目录范围重试。"
    except OSError as exc:
        return f"[错误] 无法启动 pytest：{type(exc).__name__}: {exc}"

    output = (completed.stdout or "") + (completed.stderr or "")
    summary = _summarize_pytest(output)
    summary.append(f"\n（原始输出共 {len(output)} 字符，已精简；退出码 {completed.returncode}）")
    return "\n".join(summary)


def _summarize_pytest(output: str) -> list[str]:
    """把 pytest 输出压成几行摘要。

    为什么值得单独写这个函数：
        完整输出里 90% 是 traceback 和空行，模型读完既费钱又容易被
        大段堆栈带偏注意力。而"7 passed, 2 failed"这行才是决策依据。
    """
    lines = []

    # 最后一行统计，例如 "43 passed, 2 failed in 12.34s"
    stats = [ln for ln in output.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    if stats:
        lines.append("结果：" + stats[-1].strip())

    # 失败的用例名（FAILED tests/xxx.py::test_yyy - AssertionError: ...）
    failed = re.findall(r"^FAILED\s+(\S+)(?:\s+-\s+(.*))?$", output, re.MULTILINE)
    if failed:
        lines.append(f"失败 {len(failed)} 条：")
        for name, reason in failed[:15]:
            reason = (reason or "").strip()
            if len(reason) > 160:
                reason = reason[:157] + "..."
            lines.append(f"  - {name}" + (f"  |  {reason}" if reason else ""))
        if len(failed) > 15:
            lines.append(f"  ...（还有 {len(failed) - 15} 条）")

    # 报错类型统计：AssertionError 3 次、TimeoutError 1 次...
    errors = re.findall(r"^E\s+(\w*(?:Error|Exception|Timeout))\b", output, re.MULTILINE)
    if errors:
        counted: dict[str, int] = {}
        for name in errors:
            counted[name] = counted.get(name, 0) + 1
        top = sorted(counted.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(
            "错误类型：" + "，".join(f"{name} x{count}" for name, count in top[:8])
        )

    if not lines:
        lines.append("（未从输出中解析出结论，原始输出如下）")
        lines.append(output[:3000])

    return lines


@registry.tool
def analyze_failure(log_text: str) -> str:
    """分析测试失败日志，判断失败原因并给出处理建议。

    当前是**规则版**：用关键词匹配把失败归类。
    Day 17 会把这里升级成 LLM 版（能读懂上下文、定位到具体缺陷），
    但规则层不会删 —— 因为常见的失败模式就那么几种，
    用规则判断零成本、零延迟、结果稳定，没必要每次都花钱问模型。
    这正是 Day 11 定下的原则：能用规则解决的，不要问 LLM。

    Args:
        log_text: pytest 的失败输出或错误日志文本
    """
    if not log_text.strip():
        return "[错误] 日志内容为空，无法分析"

    rules: list[tuple[str, list[str], str, str]] = [
        (
            "环境问题",
            ["connectionerror", "connection refused", "max retries", "econnreset",
             "network", "连接失败", "无法连接", "拒绝连接"],
            "测试环境没起来或网络不通，业务代码大概率没问题",
            "先确认被测服务/浏览器能正常访问，再重跑一次",
        ),
        (
            "元素定位失败",
            ["timeout", "waiting for selector", "waiting for locator",
             "no such element", "elementnotfound", "超时"],
            "页面结构变了，或者元素加载慢于等待时间",
            "确认选择器是否需要更新；先排除页面改版，再考虑加显式等待",
        ),
        (
            "服务端错误",
            ["500 internal", "internal server error", "502", "503",
             "traceback (most recent call last)", "服务端异常"],
            "服务端抛异常，属于真实缺陷，优先级高",
            "保留完整请求参数和响应，提缺陷单时一并附上",
        ),
        (
            "断言失败",
            ["assertionerror", "assert ", "expected", "actual", "断言"],
            "实际结果和预期不一致 —— 可能是缺陷，也可能是用例预期写错了",
            "先核对预期结果是否还适用，确认无误再提缺陷",
        ),
        (
            "数据问题",
            ["keyerror", "fixture", "not found in data", "数据", "fixture"],
            "测试数据缺失或结构与用例不匹配",
            "检查数据文件，确认字段齐全、未被上一次运行污染",
        ),
    ]

    lowered = log_text.lower()
    matched: list[tuple[str, str, str, int]] = []
    for category, keywords, cause, advice in rules:
        hits = sum(lowered.count(k.lower()) for k in keywords)
        if hits:
            matched.append((category, cause, advice, hits))

    lines = [f"失败日志分析结果（共 {len(log_text)} 字符）："]

    if not matched:
        lines.append("  未匹配到已知失败模式。")
        lines.append("  建议：把完整 traceback 交给人工或交给 LLM 进一步分析。")
        return "\n".join(lines)

    matched.sort(key=lambda item: item[3], reverse=True)
    for category, cause, advice, hits in matched:
        lines.append(f"\n  【{category}】关键词命中 {hits} 次")
        lines.append(f"      判断：{cause}")
        lines.append(f"      建议：{advice}")

    lines.append(f"\n  最可能的类别：{matched[0][0]}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 系统提示词
# ----------------------------------------------------------------------
def build_system_prompt() -> str:
    """给带工具的 Agent 用的系统提示词。

    和生成用例的 prompt 不同，这里**不要**规定输出格式。
    工具调用场景下模型要自由地决定"要不要调工具、调哪个"，
    逼它输出 JSON 反而会让它不敢调工具。
    """
    return (
        "你是一个测试工程助手，可以调用工具来完成任务。\n\n"
        "工作原则：\n"
        "1. 需要了解项目里有什么文件、有什么内容时，用工具去看，不要凭空猜测。\n"
        "2. 需要生成测试用例时，用 generate_testcase 工具，不要自己手写用例 JSON。\n"
        "3. 用户要保存时再调用 save_yaml；不要自作主张落盘。\n"
        "4. 工具返回错误时，先读错误信息，改正参数重试一次；"
        "还是不行就如实告诉用户。\n"
        "5. 回答用中文，简洁，说清楚你做了什么、结果是什么。\n\n"
        "可用工具：\n" + registry.describe()
    )


# ----------------------------------------------------------------------
# 自测入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 本 Agent 可用的工具 ===\n")
    print(registry.describe())
    print(f"\n共 {len(registry)} 个工具\n")
    print("=== 生成的 JSON Schema（节选）===")
    print(json.dumps(registry.schemas()[0], ensure_ascii=False, indent=2))
