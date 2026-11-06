"""从源码读取页面方法签名，不导入目标项目。"""

from __future__ import annotations

import ast
from inspect import Parameter, Signature

from agent.project_profile import ProjectProfile


def load_signatures(profile: ProjectProfile, module: str, class_name: str) -> dict[str, Signature]:
    classes: dict[str, ast.ClassDef] = {}
    paths = [profile.pages_root / profile.base_page_file,
             profile.pages_root / (module.rsplit(".", 1)[-1] + ".py")]
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        classes.update({node.name: node for node in tree.body if isinstance(node, ast.ClassDef)})

    def collect(name: str, seen: set[str]) -> dict[str, Signature]:
        if name in seen or name not in classes:
            return {}
        seen = seen | {name}
        node = classes[name]
        signatures: dict[str, Signature] = {}
        # 左侧基类的方法优先，子类覆盖继承方法。
        for base in reversed(node.bases):
            if isinstance(base, ast.Name):
                signatures.update(collect(base.id, seen))
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signatures[method.name] = _signature(method)
        return signatures

    return collect(class_name, set())


def _signature(method: ast.FunctionDef | ast.AsyncFunctionDef) -> Signature:
    args = method.args
    positional = [*args.posonlyargs, *args.args]
    default_start = len(positional) - len(args.defaults)
    static = any(isinstance(item, ast.Name) and item.id == "staticmethod"
                 for item in method.decorator_list)
    parameters = []
    for index, argument in enumerate(positional):
        if index == 0 and not static:
            continue
        kind = Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs) else Parameter.POSITIONAL_OR_KEYWORD
        default = None if index >= default_start else Parameter.empty
        parameters.append(Parameter(argument.arg, kind, default=default))
    if args.vararg:
        parameters.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(Parameter(argument.arg, Parameter.KEYWORD_ONLY,
                                    default=None if default is not None else Parameter.empty))
    if args.kwarg:
        parameters.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))
    return Signature(parameters)


def argument_issues(tree: ast.AST, page_class: str, signatures: dict[str, Signature]) -> list[str]:
    issues = []
    # 各函数单独记录页面变量，避免不同用例恰好同名时串在一起。
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for function in functions:
        variables = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == page_class:
                    variables.update(target.id for target in node.targets if isinstance(target, ast.Name))
        for call in ast.walk(function):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            owner = call.func.value
            if not isinstance(owner, ast.Name) or owner.id not in variables:
                continue
            signature = signatures.get(call.func.attr)
            if signature is None:
                continue
            # *args / **kwargs 的运行时长度未知，不能静态判定。
            if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
                continue
            try:
                signature.bind(*[None for _ in call.args], **{k.arg: None for k in call.keywords})
            except TypeError as exc:
                issues.append(f"{page_class}.{call.func.attr} 参数不匹配（第 {call.lineno} 行）：{exc}")
    return issues
