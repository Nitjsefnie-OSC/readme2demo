"""Rewrite a chained findings step so only its final segment is tolerated (#106).

``commands.sh`` makes a findings step tolerant by appending ``|| true``. On a
chained step that leaves the earlier segments unprotected: ``bash`` exempts
every command in an ``&&`` list except the last from ``set -e``, so
``cd build && tfdrift scan || true`` runs the script on past a ``cd`` that
failed, and the scan that never happened is published as verified. Two edits
fix it — guard each segment the next one depends on (``cd build || exit $?``),
and confine the tolerance to the final segment (``{ tfdrift scan || true; }``).
A segment followed by ``;`` needs no guard (``set -e`` already aborts on it),
and one followed by ``||`` must not be guarded: that operator is the author
saying the failure is handled.

Both edits need to know where the top-level segments are, and this module
refuses to guess. It recognises one shape: a chain of **simple commands**
joined by ``&&`` / ``||`` / ``;``. Every other structure — a pipeline, a
here-string, ``[[``, a grouping, a background ``&``, a compound command, a
reserved-word prefix, quoting it cannot resolve — yields ``None``, and the
caller must then emit what it emitted before this module existed. A structure
we cannot split is one we must not act on: the rewrite has nowhere safe to put
either edit, and guessing masks a failure the caller used to catch or drops
tolerance the caller used to give.
"""

from __future__ import annotations

import re

__all__ = ["tolerate_chain_tail"]

# Words that only ever introduce a construct this module refuses to model. A
# segment whose command word is one of these is declined outright, which is
# what keeps `if a; then b; fi` from being read as a three-link chain.
_RESERVED_COMMAND_WORDS = frozenset(
    "! case coproc do done elif else esac fi for function if in select then"
    " time until while".split()
)

_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\+?=")

# Characters that can never appear unquoted in a simple command we will touch:
# `(){}` open groupings and compounds, `#` opens a comment, and a newline is a
# list terminator we deliberately leave to the caller's plain append.
_FORBIDDEN = "(){}#\n\r"


def _skip_quoted(text: str, i: int, closer: str) -> int | None:
    """Index just past the quote or backtick opened at ``i``; ``None`` if open.

    Single quotes take no escapes; double quotes, ``$'…'`` and backticks do.
    """
    escapes = closer != "'"
    i += 1
    while i < len(text):
        if escapes and text[i] == "\\":
            i += 2
            continue
        if text[i] == closer:
            return i + 1
        i += 1
    return None


def _skip_balanced(text: str, i: int, opener: str, closer: str) -> int | None:
    """Index just past the ``opener``/``closer`` pair starting at ``i``.

    Quotes inside are honoured, so ``$(echo ')')`` stays balanced.
    """
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "'\"":
            nxt = _skip_quoted(text, i, ch)
            if nxt is None:
                return None
            i = nxt
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _command_word(segment: str) -> str | None:
    """The segment's command word, ignoring leading assignments.

    A quoted or escaped word is never a reserved word, so it is reported as
    ``""`` — present, but not reserved.
    """
    for raw in segment.split():
        if _ASSIGNMENT_RE.match(raw):
            continue
        return "" if any(c in raw for c in "'\"\\$`") else raw
    return None


def _split(cmd: str) -> tuple[list[tuple[int, int]], list[str], bool] | None:
    """Top-level segment spans, the operator after each, and a trailing-``;`` flag.

    ``None`` means ``cmd`` is not a chain of simple commands, or is not a shape
    this module is willing to reason about.
    """
    spans: list[tuple[int, int]] = []
    ops: list[str] = []
    start = i = 0
    n = len(cmd)

    def close(end: int) -> bool:
        text = cmd[start:end]
        stripped = text.strip()
        if not stripped:
            return False
        lead = len(text) - len(text.lstrip())
        spans.append((start + lead, start + lead + len(stripped)))
        return True

    while i < n:
        ch = cmd[i]
        if ch == "\\":
            if i + 1 >= n:
                return None  # a dangling continuation is a line we cannot see
            i += 2
            continue
        if ch in "'\"`":
            nxt = _skip_quoted(cmd, i, ch)
            if nxt is None:
                return None
            i = nxt
            continue
        if ch == "$" and cmd[i + 1 : i + 2] in ("(", "{", "'", '"'):
            opener = cmd[i + 1]
            nxt = (
                _skip_quoted(cmd, i + 1, opener)
                if opener in "'\""
                else _skip_balanced(cmd, i + 1, opener, ")" if opener == "(" else "}")
            )
            if nxt is None:
                return None
            i = nxt
            continue
        if ch in _FORBIDDEN:
            return None
        if ch == "[" and cmd[i + 1 : i + 2] == "[":
            return None  # a conditional expression, not a simple command
        if ch == "<" and cmd[i + 1 : i + 2] == "<":
            return None  # heredoc or here-string
        if ch in "|&;":
            after = cmd[i + 1 : i + 2]
            if ch == "&" and after != "&":
                if after == ">" or cmd[:i].rstrip()[-1:] in ("<", ">"):
                    i += 1  # a redirection (`&>`, `2>&1`), not an operator
                    continue
                return None  # background: the segment's status is not its own
            if ch == "|" and after != "|":
                return None  # a pipeline is one status unit we cannot split
            if ch == ";" and after in (";", "&"):
                return None
            if not close(i):
                return None
            op = ch if ch == ";" else ch * 2
            ops.append(op)
            i += len(op)
            start = i
            continue
        i += 1

    trailing = False
    if cmd[start:].strip():
        if not close(n):
            return None
        ops.append("")
    elif spans and ops[-1] == ";":
        trailing = True  # a well-formed list may end in its terminator
    else:
        return None

    for lo, hi in spans:
        segment = cmd[lo:hi]
        if segment[-1:] in ("<", ">"):
            return None  # an incomplete redirection cannot precede `||`
        if _command_word(segment) in _RESERVED_COMMAND_WORDS:
            return None
    return spans, ops, trailing


def tolerate_chain_tail(cmd: str, suffix: str = " || true") -> str | None:
    """Guard ``cmd``'s earlier segments and apply ``suffix`` to only its last.

    ``cd build && tfdrift scan`` becomes
    ``cd build || exit $? && { tfdrift scan || true; }``: a failing ``cd build``
    now aborts, and the tolerance no longer reaches past the scan (#106).
    Everything outside the rewritten segments is copied byte-for-byte.

    Returns ``None`` when the rewrite does not apply: when ``cmd`` is not a
    chain of simple commands, or when it is a single command with nothing
    before it to protect — there the caller's plain append is already correct.
    """
    split = _split(cmd)
    if split is None:
        return None
    spans, ops, trailing = split
    if len(spans) < 2 and not trailing:
        return None
    parts: list[str] = []
    pos = 0
    last = len(spans) - 1
    for idx, (lo, hi) in enumerate(spans):
        segment = cmd[lo:hi]
        if idx == last:
            segment = f"{{ {segment}{suffix}; }}"
        elif ops[idx] == "&&":
            segment = f"{segment} || exit $?"
        parts += [cmd[pos:lo], segment]
        pos = hi
    parts.append(cmd[pos:])
    return "".join(parts)
