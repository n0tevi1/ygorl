"""Conservative static script documents for the battle text comparison.

This does not execute Lua or assert that a declared effect is currently active.
The complete canonical token stream is retained alongside the setter index so
conditions, clones and helper calls are not silently reduced to bags of flags.
"""

from __future__ import annotations

import json
import re

from ygorl.build import lua

SETTER_LABELS = {
    "SetType": "effect type",
    "SetCode": "effect code or event timing",
    "SetRange": "handler location range",
    "SetTargetRange": "own and opponent target ranges",
    "SetProperty": "properties",
    "SetCondition": "activation or applicability condition callback",
    "SetTarget": "target selection or applicability callback",
    "SetCost": "cost callback",
    "SetOperation": "resolution operation callback",
    "SetValue": "value or value callback",
    "SetReset": "reset conditions",
    "SetCountLimit": "usage count limit",
}


def canonical(tokens: list[lua.Token]) -> str:
    return " ".join(json.dumps(value) if kind == "string" else value for kind, value in tokens)


def script_document(name: str, source: str | None) -> dict:
    """Return lossless token evidence and indexed setters; never infer outcomes."""
    if source is None:
        return {
            "name": name,
            "status": "missing_script",
            "setters": [],
            "canonical_source": "",
            "text": f"Card: {name}. No individual script file found. Effect semantics unknown from this source.",
        }
    script = lua.Script.parse(source)
    tokens = script.tokens
    records = []
    for i in range(len(tokens) - 3):
        if tokens[i][0] != "name" or tokens[i + 1] != ("op", ":"):
            continue
        setter = tokens[i + 2][1]
        if setter not in SETTER_LABELS:
            continue
        _, end = lua.parse_args(tokens, i + 3)
        records.append(
            {
                "function": script.function_at(i),
                "receiver": tokens[i][1],
                "setter": setter,
                "meaning": SETTER_LABELS[setter],
                "expression": canonical(tokens[i + 3 : end]),
            }
        )
    constants = sorted({value for kind, value in tokens if kind == "name" and re.fullmatch(r"[A-Z][A-Z_0-9]+", value)})
    code = canonical(tokens)
    # Symbol expansion is mechanical, not a hand-written battle-family oracle.
    symbols = "; ".join(f"{value}: {value.lower().replace('_', ' ')}" for value in constants)
    text = (
        f"Card: {name}. Static script definitions, not assertions of currently active effects. "
        "Callbacks and external helpers retain their code semantics; external helpers are not expanded.\n"
        + "\n".join(
            f"Function {r['function']}; effect variable {r['receiver']}; {r['meaning']}: {r['expression']}"
            for r in records
        )
        + f"\nSymbols: {symbols}\nComplete canonical script, including conditions and callbacks:\n{code}"
    )
    return {"name": name, "status": "parsed", "setters": records, "canonical_source": code, "text": text}
