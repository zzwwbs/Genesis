"""Error types shared across GENESIS, kept free of other GENESIS imports."""

from __future__ import annotations


class NotFoundError(KeyError):
    """A named resource -- a run, study, profile, session -- does not exist.

    The API reported every KeyError as 404 NOT_FOUND, so a dict-key fault inside
    a route read as a missing resource: a build whose processes.json lacked a
    process id answered "resource not found", and the client was told to list a
    collection that was fine. A lookup that means "no such thing" raises this;
    any other KeyError is an internal fault and is reported as one. It stays a
    KeyError, so every caller that catches KeyError to detect absence still does.
    """
