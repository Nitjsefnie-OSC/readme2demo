"""Unit tests for the top-level shell-chain rewriter (#106).

The splitter answers exactly one question: does this command parse as a
top-level chain of SIMPLE commands joined by ``&&`` / ``||`` / ``;``? When it
does, ``tolerate_chain_tail`` guards the segments the tail depends on and
confines the tolerance to the tail. When it does not, the answer is ``None``
and the caller must fall back to the series-base rendering — a structure we
cannot split is one we must not act on.
"""

from __future__ import annotations

import subprocess

import pytest

from readme2demo.shell_chain import tolerate_chain_tail


def _parses(text: str) -> bool:
    """True if bash accepts ``text`` (``bash -n``)."""
    return subprocess.run(
        ["bash", "-n"], input=text, text=True, capture_output=True
    ).returncode == 0


# -- chains the splitter must rewrite -----------------------------------------


@pytest.mark.parametrize(
    "cmd,expected",
    [
        # Issue #106's own example.
        (
            "cd build && tfdrift scan",
            "cd build || exit $? && { tfdrift scan || true; }",
        ),
        # `;` already aborts under `set -e`; only the tolerance moves.
        ("cd build; tfdrift scan", "cd build; { tfdrift scan || true; }"),
        # `||` is the author handling the failure — guarding it would break
        # the recovery the operator exists to express.
        ("a || tfdrift scan", "a || { tfdrift scan || true; }"),
        (
            "cd a && cd b && tfdrift scan",
            "cd a || exit $? && cd b || exit $? && { tfdrift scan || true; }",
        ),
        ("a || b && ftool", "a || b || exit $? && { ftool || true; }"),
        # Mixed separators keep their spelling and spacing byte-for-byte.
        ("a;b && c", "a;b || exit $? && { c || true; }"),
        ("a   &&   c", "a || exit $?   &&   { c || true; }"),
        # Redirections are part of a simple command.
        (
            "cd build && tfdrift scan 2>&1",
            "cd build || exit $? && { tfdrift scan 2>&1 || true; }",
        ),
        (
            "cd b && ftool >out.txt",
            "cd b || exit $? && { ftool >out.txt || true; }",
        ),
        (
            "cd b && ftool &>out.txt",
            "cd b || exit $? && { ftool &>out.txt || true; }",
        ),
        ("cd b && ftool >&2", "cd b || exit $? && { ftool >&2 || true; }"),
        ("cd b 2>&1 && ftool", "cd b 2>&1 || exit $? && { ftool || true; }"),
        # Quoting hides the separators inside it.
        (
            "printf '%s' 'a; b' && ftool",
            "printf '%s' 'a; b' || exit $? && { ftool || true; }",
        ),
        (
            'echo "x && y" && ftool',
            'echo "x && y" || exit $? && { ftool || true; }',
        ),
        ("echo $'a;b' && ftool", "echo $'a;b' || exit $? && { ftool || true; }"),
        # Substitutions are opaque: a separator inside one is not top level.
        (
            "echo $(a; b) && ftool",
            "echo $(a; b) || exit $? && { ftool || true; }",
        ),
        ("echo `a; b` && ftool", "echo `a; b` || exit $? && { ftool || true; }"),
        (
            "echo ${x:-a;b} && ftool",
            "echo ${x:-a;b} || exit $? && { ftool || true; }",
        ),
        ("echo \\; && ftool", "echo \\; || exit $? && { ftool || true; }"),
        # A trailing terminator is part of a well-formed list. Rewriting a
        # single segment here is a strict improvement: the base rendering
        # ("ftool; || true") is not valid bash at all.
        ("ftool;", "{ ftool || true; };"),
        ("cd b; ftool ;", "cd b; { ftool || true; } ;"),
        # Assignment-only and assignment-prefixed segments are simple commands.
        ("FOO=1 && ftool", "FOO=1 || exit $? && { ftool || true; }"),
        ("cd b && FOO=1 ftool", "cd b || exit $? && { FOO=1 ftool || true; }"),
    ],
)
def test_rewrites_top_level_chain(cmd: str, expected: str) -> None:
    assert tolerate_chain_tail(cmd) == expected
    assert _parses(expected)


# -- structures the splitter must decline -------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        # A lone simple command needs no isolation; the base rendering
        # ("cmd || true") is already correct and is what the caller emits.
        "tfdrift scan",
        "tfdrift scan 2>&1",
        # Pipelines.
        "tfdrift scan | tee drift.txt",
        "cd build && tfdrift scan | tee drift.txt",
        "sh -c 'exit 3' | cat ; ftool",
        "a |& b",
        # Here-strings and heredocs.
        "ftool <<< 'x'",
        "cd b && ftool <<< 'x'",
        "cat <<EOF\nbody\nEOF",
        # Conditionals, arithmetic, groupings, subshells.
        "[[ -f nope ]] ; ftool",
        "cd b && [[ -f nope ]]",
        "(( 1 + 1 )) ; ftool",
        "( sh -c 'exit 3' ) ; ftool",
        "{ sh -c 'exit 3'; } ; ftool",
        "cd b && ( ftool )",
        # Compound commands.
        "if true; then ftool; fi",
        "for i in 1; do ftool; done ; x",
        "while true; do ftool; done",
        "until false; do ftool; done",
        "case x in y) ftool;; esac",
        "select x in a; do ftool; done",
        "function f { ftool; }",
        # Reserved words with no command word (round-7 defect 1's family).
        "time --",
        "time -p",
        "! ftool",
        "time ftool",
        "cd b && time ftool",
        "cd b && ! ftool",
        # Background.
        "ftool &",
        "a & ftool",
        # Newlines and comments.
        "a\nftool",
        "ftool # scan",
        "a && ftool # scan",
        # Quoting we cannot resolve.
        "echo 'unterminated && ftool",
        'echo "unterminated && ftool',
        "echo $(unbalanced && ftool",
        "ftool \\",
        # Degenerate lists.
        "",
        "   ",
        ";",
        "; ftool",
        "a && && b",
        "a ;; b",
    ],
)
def test_declines_structures_it_cannot_split(cmd: str) -> None:
    assert tolerate_chain_tail(cmd) is None


def test_rewrite_preserves_separator_spelling_and_spacing() -> None:
    """Only the segments are touched; the operators between them are copied."""
    cmd = "cd  build   &&\tmake -j2  ;   tfdrift scan --path /tmp/x"
    got = tolerate_chain_tail(cmd)
    assert got == (
        "cd  build || exit $?   &&\tmake -j2  ;   "
        "{ tfdrift scan --path /tmp/x || true; }"
    )
    assert _parses(got)


def test_custom_suffix_is_honoured() -> None:
    assert (
        tolerate_chain_tail("a && b", suffix=" || :")
        == "a || exit $? && { b || :; }"
    )


@pytest.mark.parametrize(
    "cmd,failing_prefix,expect_abort",
    [
        # #106: a failing prerequisite must not be published as verified.
        ("{p} && ftool", "sh -c 'exit 7'", True),
        ("{p} && x && ftool", "sh -c 'exit 7'", True),
        ("x && {p} && ftool", "sh -c 'exit 7'", True),
        ("{p}; ftool", "sh -c 'exit 7'", True),
        # A `||` prefix is a handled failure: the tail is the recovery.
        ("{p} || ftool", "sh -c 'exit 7'", False),
        # A succeeding prefix must let the tolerated tail through.
        ("{p} && ftool", "true", False),
        ("{p}; ftool", "true", False),
        ("{p} || ftool", "true", False),
    ],
)
def test_rewritten_chain_behaves_under_set_e(
    cmd: str, failing_prefix: str, expect_abort: bool
) -> None:
    """Execute the rewrite — matching its text proves nothing about ``set -e``."""
    step = cmd.format(p=failing_prefix).replace("ftool", "sh -c 'exit 3'")
    rewritten = tolerate_chain_tail(step)
    assert rewritten is not None
    proc = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{rewritten}\necho REACHED"],
        capture_output=True,
        text=True,
    )
    assert ("REACHED" not in proc.stdout) is expect_abort, (rewritten, proc)
