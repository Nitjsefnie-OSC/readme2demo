"""Property tests for findings-step tolerance over a generated corpus (#106).

``commands.sh`` runs under ``set -euxo pipefail``. A findings tool (drift
detector, linter, scanner) exits nonzero ON SUCCESS, so its step is made
tolerant. Issue #106: appending ``|| true`` to a *chained* step also swallows
the failure of every earlier segment, so a broken setup step is published as
verified.

Two rules are pinned here, in BOTH marking modes (the whole step marked, and
only the chain tail marked):

1. When the step parses as a top-level chain of simple commands, only the final
   segment is tolerated and an earlier segment's failure still aborts.
2. When it does not parse, the emission is byte-identical to the series base —
   a structure we cannot split is one we must not act on. The base rule is
   restated here directly (append ``|| true`` to the whole command when the
   whole command or its naively-split last segment is findings-marked) so the
   floor is asserted, not assumed.

Every runtime case is EXECUTED. Round after round of `bash -n`-and-tolerance
tables passed while the runtime behaviour was broken; matching rewrite text is
not evidence about ``set -e``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from readme2demo.distill import normalize_cmd, write_commands_sh
from readme2demo.normalize import mark_findings_success
from readme2demo.types import (
    AgentResult,
    CommandEntry,
    CommandLog,
    DistillOutput,
    Plan,
    SuccessCriteria,
    TutorialOutline,
)

PATTERN = "DRIFT"

_BASE_CHAIN_SPLIT_RE = re.compile(r"\s*(?:&&|;)\s*")


@pytest.fixture(scope="session")
def ftool(tmp_path_factory) -> str:
    """A real findings tool: prints what the demo shows, then exits nonzero."""
    path = tmp_path_factory.mktemp("bin") / "ftool"
    path.write_text("#!/bin/sh\necho DRIFT\nexit 3\n")
    path.chmod(0o755)
    return str(path)


# Prefix segments the structural splitter can resolve (simple commands), each
# in a failing and a succeeding spelling.
SIMPLE_PREFIXES = [
    ("sh -c 'exit 3'", "true"),
    ("test -f /nonexistent-r2d-106", "test -d /"),
    ("env FOO=1 sh -c 'exit 4'", "env FOO=1 true"),
]

# Prefix segments it cannot resolve — pipelines, conditionals, groupings,
# here-strings, compounds. These must fall back to the base emission.
DECLINED_PREFIXES = [
    ("sh -c 'exit 3' | cat", "true | cat"),
    ("[[ -f /nonexistent-r2d-106 ]]", "[[ -d / ]]"),
    ("( sh -c 'exit 3' )", "( true )"),
    ("{ sh -c 'exit 3'; }", "{ true; }"),
    ("sh -c 'exit 3' <<< x", "cat <<< x"),
    ("for i in 1; do sh -c 'exit 3'; done", "for i in 1; do true; done"),
    ("! true", "! false"),
]

# Tail shapes, again split by whether the splitter can resolve them.
SIMPLE_TAILS = ["{f} --path /tmp 2>&1"]
DECLINED_TAILS = ["{f} | cat"]

SEPARATORS = ["&&", ";"]
MODES = ["whole", "tail"]


def _nested_expansion_cases() -> list[tuple[str, str]]:
    """Generate chains whose quoted expansion data contains shell operators."""
    expansions = [
        '$(printf "%s && %s" left right)',
        '$(printf "%s; %s" left right)',
        '${unset:-"left && right"}',
        '${unset:-"left; right"}',
        '$(printf "%s && marker" x)',
        '`printf "%s && %s" left right`',
        '$(printf "%s" "$(printf "%s && %s" left right)")',
        '${unset:-"$(printf "%s && %s" left right)"}',
        '$[1 && 2]',
    ]
    wrappers = [
        'echo "{exp}"',
        'printf "%s" "{exp}"',
    ]
    tails = ["r2d_findings --scan", "r2d_findings --scan --json"]
    separators = ["&&", ";"]
    return [
        (
            f'{wrapper.format(exp=exp)} {separator} {tail}',
            (
                f'{wrapper.format(exp=exp)} || exit $? && {{ {tail} || true; }}'
                if separator == "&&"
                else f'{wrapper.format(exp=exp)} ; {{ {tail} || true; }}'
            ),
        )
        for exp in expansions
        for wrapper in wrappers
        for separator in separators
        for tail in tails
    ]


CORPUS = [
    (pkind, failing, ok, tkind, tail, sep, mode)
    for pkind, prefixes in (
        ("simple", SIMPLE_PREFIXES),
        ("declined", DECLINED_PREFIXES),
    )
    for failing, ok in prefixes
    for tkind, tails in (("simple", SIMPLE_TAILS), ("declined", DECLINED_TAILS))
    for tail in tails
    for sep in SEPARATORS
    for mode in MODES
]


def _base_emission(cmd: str, findings: set[str]) -> str:
    """What the series base emitted for ``cmd`` — the floor for rule 2."""
    norm = normalize_cmd(cmd)
    last_seg = _BASE_CHAIN_SPLIT_RE.split(norm)[-1].strip()
    if (norm in findings or last_seg in findings) and "|| true" not in cmd:
        return f"{cmd} || true"
    return cmd


def _findings_set(log: CommandLog) -> set[str]:
    """The base's findings key set, rebuilt from the marked log."""
    found = {normalize_cmd(e.cmd) for e in log.entries if e.findings_success}
    found |= {c.split("|", 1)[0].strip() for c in found}
    return found


def render(step: str, tail: str, mode: str, tmp_path: Path) -> tuple[str, str, set[str]]:
    """Render commands.sh for a one-step run through the production path.

    Returns ``(script, emitted_step_line, findings_keys)``. ``mode`` selects
    which log entry is findings-marked: the whole ``step`` or just its ``tail``.
    """
    marked = step if mode == "whole" else tail
    log = CommandLog(
        engine="claude-code",
        entries=[CommandEntry(cmd=marked, exit_code=3, output="DRIFT detected")],
        result=AgentResult(outcome="success"),
    )
    plan = Plan(
        quickstart_summary="q",
        success_criteria=SuccessCriteria(command=marked, expected_pattern=PATTERN),
    )
    assert mark_findings_success(plan, log) == 1, "the entry must really be marked"
    out = DistillOutput(
        commands=[step], tape=[], outline=TutorialOutline(title="T", intro="I.")
    )
    run_dir = tmp_path / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    write_commands_sh(out, run_dir, plan, "", log)
    script = (run_dir / "commands.sh").read_text()
    body = script.split("cd /work\n", 1)[1].split("\n# --- readme2demo success", 1)[0]
    return script, body.strip("\n"), _findings_set(log)


def execute(script: str, tmp_path: Path) -> tuple[int, bool, str]:
    """Run the rendered script in a scratch dir.

    Returns ``(rc, published, stderr)``. Bash refuses to run a script it cannot
    parse, so ``stderr`` carries the parse verdict as well — no separate
    ``bash -n`` pass needed, and the evidence is what actually ran.
    """
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    proc = subprocess.run(
        ["bash", "-c", script.replace("cd /work", f"cd {work}", 1)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, "R2D_VERIFY_OK" in proc.stdout, proc.stderr


def parses(script: str) -> bool:
    return subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True
    ).returncode == 0


CASE = "pkind,failing,ok,tkind,tail,sep,mode"


def test_every_corpus_atom_is_valid_bash(ftool):
    """The corpus must be made of real commands, or nothing below means much.

    Joining two valid complete commands with ``&&`` / ``;`` yields valid bash,
    so validating the atoms validates every generated step.
    """
    atoms = [p for pair in SIMPLE_PREFIXES + DECLINED_PREFIXES for p in pair]
    atoms += [t.format(f=ftool) for t in SIMPLE_TAILS + DECLINED_TAILS]
    bad = [a for a in atoms if not parses(a)]
    assert bad == []


@pytest.mark.parametrize(CASE, CORPUS)
def test_findings_chain_properties(
    pkind, failing, ok, tkind, tail, sep, mode, tmp_path, ftool
):
    """All five properties for one corpus case, executed.

    (A) the rendered script parses; (B) a findings tail never aborts the run;
    (C) a failing earlier segment is never masked — for ``;`` the base already
    aborted, and for ``&&`` this is #106's fix, applied exactly where the step
    splits into simple commands; (D) rule 2 — an unsplittable structure renders
    byte-identically to the series base; (E) tolerance is never less than base.
    """
    tail = tail.format(f=ftool)
    splittable = pkind == "simple" and tkind == "simple"
    # `set -e` deliberately exempts a `!`-inverted command, so that prefix's
    # nonzero status is not a failure the script is supposed to abort on.
    aborts = not failing.lstrip().startswith("!")

    for prefix, prefix_succeeds in ((failing, False), (ok, True)):
        step = f"{prefix} {sep} {tail}"
        script, emitted, findings = render(step, tail, mode, tmp_path)
        base = _base_emission(step, findings)

        if not splittable:
            assert emitted == base, f"(D) rule 2 violated for {step!r}"
        if base != step:
            assert emitted != step, f"(E) base tolerated this, HEAD did not: {step}"

        rc, published, err = execute(script, tmp_path)
        assert "syntax error" not in err, f"(A) unparseable render\n{script}\n{err}"
        if prefix_succeeds:
            assert rc == 0 and published, f"(B) findings tail aborted\n{script}"
        elif aborts and (sep == ";" or splittable):
            assert rc != 0, f"(C) prefix failure masked\n{script}"
            assert not published, f"(C) a failed step was published\n{script}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", _nested_expansion_cases())
def test_nested_expansion_is_opaque_in_rendered_findings_steps(case, mode, tmp_path):
    """Nested expansion data stays byte-identical while the outer chain is isolated."""
    step, expected = case
    tail = re.search(r"r2d_findings --scan(?: --json)?$", step).group(0)
    _, emitted, findings = render(step, tail, mode, tmp_path)
    assert emitted in {expected, _base_emission(step, findings)}


@pytest.mark.parametrize("mode", MODES)
def test_issue_106_headline_example(mode, tmp_path, ftool):
    """The issue's own example, in both modes: `cd build && tfdrift scan`.

    ``bash`` exempts a non-final command of an ``&&`` list from ``set -e``, so
    isolating the tolerance is necessary but not sufficient — the prerequisite
    needs its own guard or its failure still goes unnoticed.
    """
    step = f"cd /tmp && {ftool}"
    _, emitted, _ = render(step, ftool, mode, tmp_path)
    assert emitted == f"cd /tmp || exit $? && {{ {ftool} || true; }}"


@pytest.mark.parametrize("mode", MODES)
def test_or_list_tail_is_isolated(mode, tmp_path, ftool):
    """A ``||`` list splits too — base's regex never saw those separators."""
    step = f"cd /tmp || {ftool}"
    _, emitted, findings = render(step, ftool, mode, tmp_path)
    if _base_emission(step, findings) == step:
        pytest.skip("base did not tolerate this shape; rule 2 keeps it bare")
    assert emitted == f"cd /tmp || {{ {ftool} || true; }}"


@pytest.mark.parametrize("mode", MODES)
def test_failing_build_then_scan_is_not_published(mode, tmp_path, ftool):
    """Regression (round-9 adversary counterexample 1): a failed build step
    followed by a scan must not print R2D_VERIFY_OK."""
    step = f"sh -c 'exit 2' && {ftool}"
    script, _, _ = render(step, ftool, mode, tmp_path)
    rc, published, _ = execute(script, tmp_path)
    assert rc != 0 and not published, script
