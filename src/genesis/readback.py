"""Readback: say what a compiled build actually does, in the researcher's language.

Compilation answers whether a package is well formed. The intent check, at the
draft, answers whether a declaration follows from what the researcher said. This
answers the third question, and the one a reviewer will eventually ask: read
cold, with no knowledge of what was intended, what does this study do?

The reader is deliberately **blind to the design**. Shown the intent it restates
the intent; shown only the declarations it has to work out what they produce, and
says so in prose a social scientist can check against their own design without
reading YAML. A cardinality cap that keeps the three least-clicked articles is
invisible in the declaration and obvious in the sentence.

Like the intent check this is advisory and pinned: the model profile and a digest
of the request are recorded with the result, so a reading can be compared with
the one that was accepted. It is generated on request rather than at every
compile, because it costs a model call and answers a question worth asking once
a package is finished.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# What is needed to say what reaches an actor and what the conditions change.
# Nothing here identifies the study's intent; that is the point.
BUILD_PARTS = (
    "initialization.json",
    "processes.json",
    "context_policies.json",
    "state_model.json",
    "protocol.json",
    "theory_execution_plan.json",
)

# Keys that carry the researcher's own justification rather than what the
# package does. The reader is meant to be blind to the design, and these are the
# design stated in prose -- shown them, it restates the intent instead of
# working out what the declarations produce, which is the whole point of asking.
#
# Matched by suffix, not enumerated. An enumeration lost this race twice: it
# named openness_rationale and closure_rationale and missed measurement_use's
# plain `rationale` and the execution plan's `reason`, so a real build still
# handed the reader the design in prose. A new *_rationale is caught the day it
# is added rather than the next time someone reads a prompt.
INTENT_BEARING_SUFFIXES = ("rationale", "reason", "justification", "origin")


def _is_intent_bearing(key: str) -> bool:
    """Whether a declaration key holds prose about why, not a statement of what."""
    return key.endswith(INTENT_BEARING_SUFFIXES)


def _without_intent(value: Any) -> Any:
    """The same declarations with every stated rationale removed."""
    if isinstance(value, Mapping):
        return {
            key: _without_intent(item)
            for key, item in value.items()
            if not _is_intent_bearing(str(key))
        }
    if isinstance(value, list):
        return [_without_intent(item) for item in value]
    return value


SYSTEM = (
    "You read a compiled agent-based-simulation package and report what it "
    "actually does. You are a careful reader, not an author: never say what the "
    "study probably intends, only what these declarations determine. Where a "
    "declaration produces a surprising or degenerate result, say so plainly "
    "rather than smoothing it over. Write for a social scientist who does not "
    "read YAML."
)

_TASK = """Read the compiled declarations above and answer, in plain language:

1. For EVERY context policy, state exactly what an actor bound to it receives:
   which records, whose records, HOW MANY, and -- where a cardinality cap
   declares an ordering -- WHICH ones, by what ordering, and from which end.
   Work out what the ordering field resolves to on the records in question; if
   it resolves to nothing, say what the cap then does.
2. Under which conditions and at which rounds does each gated item appear?
3. Which processes are measurements, and which actors can and cannot see what
   they produce? Answer from the declarations only; the analysis plan is not
   shown to you, so do not guess at what the outcomes compute.
4. What does each experimental condition change about what actors experience?
5. List anything that appears degenerate, unreachable, inverted, or that would
   silently produce an empty or arbitrary result at run time.

Be concrete: name the policy, the field, and the consequence.

Note on the runtime, so you do not report it as a defect: every state record the
runtime writes is annotated with the phase it was written in, so a cardinality
ordering by `phase` resolves. Any other ordering field must come from the record
itself. A process reads a declared input artifact under `inputs`, not under the
artifact's own name."""


def assemble_readback_request(parts: Mapping[str, str]) -> str:
    """Compose the reading prompt from the compiled declarations alone."""
    stripped: dict[str, str] = {}
    for name, text in parts.items():
        try:
            stripped[name] = json.dumps(_without_intent(json.loads(text)), indent=2)
        except (TypeError, ValueError):
            # Not JSON, or not shaped as expected: pass it through rather than
            # dropping a declaration the reading may need.
            stripped[name] = text
    body = "\n".join(f"----- {name}\n{text}" for name, text in stripped.items()) or "(empty build)"
    return f"# Compiled package declarations\n\n{body}\n\n# Your task\n\n{_TASK}"


def readback_record(
    text: str,
    *,
    build_ref: str,
    build_hash: str,
    model_profile: str,
    request_digest: str,
) -> dict[str, Any]:
    """The stored reading, pinned to what produced it.

    A reading that cannot be tied to a build and a model is not evidence: the
    researcher accepted a description of something, and the record has to say
    of what.
    """
    return {
        "kind": "build_readback",
        "build": build_ref,
        "build_hash": build_hash,
        "model_profile": model_profile,
        "request_digest": request_digest,
        "text": text,
    }
