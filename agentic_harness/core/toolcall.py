"""Qwen3.5 / Qwen3-Coder XML tool-call format: parse and render.

Format emitted by the Qwen3.5 chat template (checkpoint chat_template.jinja):

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>

Parsing rules follow vLLM's ``qwen3_xml`` / ``qwen3_coder`` tool parser:
parameter values are raw strings and are converted using the tool's JSON schema
(integer/number/boolean/object/array/null). One parser is shared by every policy
that sees raw text (HF local, vLLM completions, ms-swift scheduler) so that
training and evaluation see identical parsing.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)


@dataclass
class ParsedCall:
    name: str
    arguments: dict
    raw: str = ""


@dataclass
class ParseResult:
    content: str                      # assistant text outside tool calls (after </think>)
    reasoning: str | None             # text inside <think>...</think> (None if no think block)
    tool_calls: list[ParsedCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    think_unclosed: bool = False


def _strip_one_newline(s: str) -> str:
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


def _convert(value: str, schema: dict | None) -> Any:
    """Convert a raw XML parameter string using the JSON schema type (vLLM qwen3 rules)."""
    if schema is None:
        # unknown parameter: try JSON, else keep string
        try:
            return json.loads(value)
        except Exception:
            return value
    t = schema.get("type")
    if t is None:
        for key in ("anyOf", "oneOf"):           # e.g. pydantic Optional[X] -> anyOf: [X, null]
            alts = [a for a in schema.get(key) or [] if isinstance(a, dict) and a.get("type") not in (None, "null")]
            if alts:
                schema, t = alts[0], alts[0].get("type")
                break
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if value.strip().lower() == "null" and t != "string":
        return None
    try:
        if t == "string" or t is None:
            return value
        if t == "integer":
            return int(value.strip())
        if t == "number":
            f = float(value.strip())
            return int(f) if f.is_integer() else f   # same as vLLM qwen3 parser (5.0 -> 5)
        if t == "boolean":
            v = value.strip().lower()
            if v in ("true", "1"):
                return True
            if v in ("false", "0"):
                return False
            return value
        if t in ("object", "array"):
            try:
                return json.loads(value)
            except Exception:
                return ast.literal_eval(value)
    except Exception:
        return value  # leave as string; Env schema validation reports the type error
    return value


def split_think(text: str, prompt_opened_think: bool) -> tuple[str | None, str, bool]:
    """Return (reasoning, content, unclosed).

    ``prompt_opened_think``: the generation prompt already ended with ``<think>\\n``
    (thinking mode), so the completion starts inside the think block.
    """
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        reasoning = reasoning.split("<think>")[-1].strip("\n")
        return reasoning, content.lstrip("\n"), False
    if prompt_opened_think:
        return text.strip("\n"), "", True
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[len("<think>"):].strip("\n"), "", True
    return None, text, False


def parse_assistant(text: str, tool_schemas: list[dict] | None = None, *, prompt_opened_think: bool = False,
                    stop_strings: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>")) -> ParseResult:
    for s in stop_strings:
        if s in text:
            text = text.split(s, 1)[0]
    reasoning, content, unclosed = split_think(text, prompt_opened_think)
    res = ParseResult(content=content, reasoning=reasoning, think_unclosed=unclosed)
    if unclosed and "<tool_call>" in (reasoning or ""):
        res.errors.append("tool_call_inside_unclosed_think")
    props = {}
    for s in tool_schemas or []:
        fn = s.get("function", s)
        props[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    first = content.find("<tool_call>")
    if first < 0:
        if "<function=" in content:
            res.errors.append("function_tag_without_tool_call")
        res.content = content.strip()
        return res
    res.content = content[:first].strip()
    for m in TOOL_CALL_RE.finditer(content[first:]):
        block = m.group(1)
        fm = FUNCTION_RE.search(block)
        if not fm:
            # hermes-style JSON fallback: {"name": ..., "arguments": {...}}
            try:
                obj = json.loads(block.strip())
                res.tool_calls.append(ParsedCall(obj["name"], dict(obj.get("arguments") or {}), m.group(0)))
                res.errors.append("json_style_tool_call")
            except Exception:
                res.errors.append("unparseable_tool_call")
            continue
        name = fm.group(1).strip()
        body = fm.group(2)
        args = {}
        schema_props = props.get(name, {})
        for pm in PARAM_RE.finditer(body):
            pname = pm.group(1).strip()
            raw = _strip_one_newline(pm.group(2))
            args[pname] = _convert(raw, schema_props.get(pname))
        res.tool_calls.append(ParsedCall(name, args, m.group(0)))
    tail = TOOL_CALL_RE.sub("", content[first:]).strip()
    if tail:
        res.errors.append("text_after_tool_call")
    return res


def _render_value(v: Any, style: str) -> str:
    """style='jinja' mirrors the checkpoint chat template; style='swift' mirrors ms-swift qwen3_5 agent template."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if style == "swift" and (isinstance(v, bool) or v is None):
        return json.dumps(v)
    return str(v)


def render_tool_calls(calls: list[dict], content: str = "", style: str = "jinja") -> str:
    """Render assistant text for tool calls exactly like the chat template would."""
    out = []
    for i, c in enumerate(calls):
        if i == 0:
            out.append(("\n\n" if content.strip() else "") + f"<tool_call>\n<function={c['name']}>\n")
        else:
            out.append(f"\n<tool_call>\n<function={c['name']}>\n")
        for k, v in (c.get("arguments") or {}).items():
            out.append(f"<parameter={k}>\n{_render_value(v, style)}\n</parameter>\n")
        out.append("</function>\n</tool_call>")
    return content + "".join(out)
