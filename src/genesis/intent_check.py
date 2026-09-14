"""Intent check: does the drafted stage say what the researcher said?

Cross-layer validation asks whether declarations agree with each other, and
compilation asks whether they are well formed. Neither can ask the question that
actually decides whether a study measures what it claims to: does this
declaration follow from what the researcher asked for? That question is not
decidable, so it belongs to a reader rather than to a rule -- and to a reader
placed where the answer is still cheap to act on, which is the draft, before
approval writes an immutable version and downstream stages build on it.

Two verdicts are reported and nothing else:

``contradicted``
    The researcher said one thing and the declaration says another. This is the
    finding worth interrupting for, so it must be acknowledged before approval.

``unsupported``
    The declaration settles something consequential the conversation never
    determined. Often a reasonable default -- but a default asserted silently is
    indistinguishable, later, from a decision the researcher made. The patch
    already carries an ``assumptions`` vocabulary for exactly this, so the
    remedy is to record it there.

The reader is the same model that drafted the package, which is what a real
deployment looks like. That correlation is a real limit: a misreading shared by
drafter and reader survives both. It does not make the check worthless -- the
draft and the check are different tasks, and the check sees the researcher's own
words next to the declaration rather than generating from them -- but findings
are advisory evidence for a researcher, never a certificate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .elicitation import _extract_json_object

VERDICTS = frozenset({"contradicted", "unsupported"})

SYSTEM = (
    "You verify that a drafted study specification says what the researcher "
    "said. You are not an author and not a scientific reviewer: never judge "
    "whether a choice is good, only whether the declarations match the stated "
    "intent. Quote the researcher. Respond with ONLY a JSON object."
)

_TASK = """For each declaration under review, decide one of:

  "contradicted" - the researcher said something and the declaration says
                   otherwise. Quote the turn you are relying on.
  "unsupported"  - the declaration settles something consequential that the
                   turns never determined. It may be a reasonable default, but
                   it should be recorded as an assumption, not asserted.

Report ONLY these two. Do not list declarations that match the turns. Do not
comment on mechanical detail the researcher would have no opinion about: field
ordering, naming style, schema shape, or the grammar a declaration is written
in. If nothing qualifies, return an empty list.

Return exactly:
{"findings": [{"verdict": "contradicted"|"unsupported",
               "declaration": "<path or id in the draft>",
               "researcher_said": "<short verbatim quote, or null>",
               "draft_says": "<what the declaration actually states>",
               "consequence": "<what happens at run time because of this>"}]}"""


def finding_id(finding: Mapping[str, Any]) -> str:
    """A stable id, so an acknowledgement survives regenerating the preview."""
    material = json.dumps(
        [
            str(finding.get("verdict", "")),
            str(finding.get("declaration", "")),
            str(finding.get("draft_says", "")),
            # The quote the finding rests on is part of what was acknowledged.
            # Without it, a finding re-grounded in a different sentence kept the
            # id of the one the researcher had already waved through, so the
            # new reasoning was never put to them.
            str(finding.get("researcher_said") or ""),
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def assemble_intent_request(
    turns: list[tuple[str, str]],
    stage_id: str,
    under_review: Mapping[str, str],
    upstream: Mapping[str, str],
) -> str:
    """Compose the check prompt from the researcher's own words and the draft.

    ``turns`` is the whole conversation, not just this stage's: intent is
    cumulative, and the declaration that contradicts it is often in a different
    layer from the sentence that stated it.
    """
    spoken = (
        "\n\n".join(
            f"### turn {index + 1} (stage: {turn_stage})\n{answer.strip()}"
            for index, (turn_stage, answer) in enumerate(turns)
            if answer and answer.strip()
        )
        or "(no researcher turns recorded)"
    )
    review = "\n".join(f"----- {name}\n{text}" for name, text in under_review.items()) or "(empty)"
    prior = "\n".join(f"----- {name}\n{text}" for name, text in upstream.items()) or "(none)"
    return (
        f"# What the researcher said, in their own words\n\n{spoken}\n\n"
        f"# Declarations under review (stage: {stage_id})\n\n{review}\n\n"
        f"# Declarations already approved in other layers\n\n{prior}\n\n"
        f"# Your task\n\n{_TASK}"
    )


def parse_intent_findings(text: str) -> list[dict[str, Any]]:
    """Read the findings object, dropping anything malformed.

    A check is advisory, so a garbled response costs the researcher nothing and
    must never fail the preview; the caller records it as unavailable instead.
    """
    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        raise ValueError("INTENT_CHECK: response is not a JSON object")
    raw = payload.get("findings")
    if not isinstance(raw, list):
        raise ValueError("INTENT_CHECK: response has no findings list")
    findings: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        declaration = str(item.get("declaration", "")).strip()
        if verdict not in VERDICTS or not declaration:
            continue
        said = item.get("researcher_said")
        finding = {
            "verdict": verdict,
            "declaration": declaration,
            "researcher_said": str(said) if said else None,
            "draft_says": str(item.get("draft_says", "")),
            "consequence": str(item.get("consequence", "")),
        }
        finding["id"] = finding_id(finding)
        findings.append(finding)
    # Contradictions first: they are the ones that block approval.
    findings.sort(key=lambda entry: (entry["verdict"] != "contradicted", entry["declaration"]))
    return findings


def blocking_findings(check: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Contradictions the researcher has not acknowledged."""
    if not isinstance(check, Mapping):
        return []
    acknowledged = {str(item) for item in check.get("acknowledged", []) or ()}
    return [
        dict(finding)
        for finding in check.get("findings", []) or ()
        if isinstance(finding, Mapping)
        and finding.get("verdict") == "contradicted"
        and str(finding.get("id")) not in acknowledged
    ]
