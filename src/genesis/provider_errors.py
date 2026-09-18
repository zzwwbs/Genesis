"""Which provider failures should pause a run rather than fail it.

Kept free of other GENESIS imports: the runtime classifies an executor's
exception with it, and the provider client, which imports the runtime, shares
its status sets.
"""

from __future__ import annotations

import re
from typing import Any

TRANSIENT_HTTP_STATUSES = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
)
# Provider responses that say the provider cannot serve the call right now -- no
# credit, a rejected or exhausted key, rate limiting or an outage that outlasted
# the retries -- as opposed to a request that is itself wrong (400, 404, 422).
PAUSING_HTTP_STATUSES = frozenset({401, 402, 403}) | TRANSIENT_HTTP_STATUSES
_PROVIDER_HTTP_STATUS = re.compile(r"^PROVIDER_HTTP: provider returned HTTP (\d{3})")
# The part of an error GENESIS composed itself, before the provider's own body:
# "PROVIDER_HTTP: provider returned HTTP 402 after 3 retries: <body>".
_OWN_WORDS = re.compile(r"^(PROVIDER_[A-Z_]+: [^:]*?(?: after \d+ retr(?:y|ies))?)(?::|$)")


def _own_words(message: str) -> str:
    """The error without the provider's response body.

    A provider routinely echoes the request it refused, so its body can carry
    the prompt. Everything after GENESIS's own prefix is dropped rather than
    truncated.
    """
    found = _OWN_WORDS.match(message)
    return (found.group(1) if found else message.split(":")[0])[:300]


def is_provider_cancellation(error: BaseException) -> bool:
    """Whether a call was aborted deliberately rather than failing.

    The pause and cancel paths both set the run's shared cancel event, and the
    in-flight provider call then raises ``PROVIDER_CANCELLED``. That is not a
    fault in the call: the researcher stopped it. Recorded as a failed attempt
    it made the run terminal, which is the opposite of what pressing Pause is
    for.
    """
    if str(error).startswith("PROVIDER_CANCELLED"):
        return True
    cause = error.__cause__
    return (
        isinstance(cause, BaseException) and cause is not error and is_provider_cancellation(cause)
    )


def provider_pause_reason(error: BaseException) -> dict[str, Any] | None:
    """Why a failed call should pause its run rather than fail it, or ``None``.

    Running out of credit or a provider outage longer than the retries is not a
    fault in the study: the run can continue once the provider can serve it. A
    failed run cannot be resumed, so treating these as failures lost the run.

    The reason is persisted on the run and returned by the API, so it carries
    only what GENESIS itself wrote. Truncating the provider's body bounded its
    length, not its content, and a provider that echoes the offending request
    echoes the prompt with it: a 402 body put the authorized context into
    ``executions[].paused_by``. The body stays on the chained exception, where
    the trace policy governs it.
    """
    message = str(error)
    if message.startswith("PROVIDER_UNAVAILABLE"):
        return {"kind": "provider_unavailable", "status": None, "error": _own_words(message)}
    found = _PROVIDER_HTTP_STATUS.match(message)
    if found and int(found.group(1)) in PAUSING_HTTP_STATUSES:
        status = int(found.group(1))
        kind = {401: "provider_auth", 402: "provider_credit", 403: "provider_auth"}.get(
            status, "provider_unavailable" if status != 429 else "provider_rate_limit"
        )
        return {"kind": kind, "status": status, "error": _own_words(message)}
    cause = error.__cause__
    if isinstance(cause, BaseException) and cause is not error:
        return provider_pause_reason(cause) if str(cause).startswith("PROVIDER_") else None
    return None
