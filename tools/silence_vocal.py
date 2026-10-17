"""Python reference for YuE2Kit's silenceVocal (Instrumental.swift). The two must stay byte-identical.

    swift: yue2cli --silence-abc list.txt        (writes <file>.swift-silenced for every path in list.txt)
    python -I tools/silence_vocal.py list.txt    (writes <file>.py-silenced and compares with the Swift output)

Standard library only. vendor/abc_tools.py (Apache-2.0, from the upstream yue2-music skill) validates the result.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ABC_TOKEN = re.compile(r'"[^"\n]*"|\[K:[^\]\n]+\]|(?:\^\^|__|\^|_|=)?([A-Ga-gz])[,\']*([0-9]*)-?')


def _merge_rests(line: str) -> str:
    """z4z4z2 -> z8z2: merge runs of rests (between chord symbols, within a bar) into the native lengths."""
    allowed = (48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)

    def run(m):
        total, out = sum(int(d or 1) for d in re.findall(r"z([0-9]*)", m.group(0))), []
        for a in allowed:
            while total >= a:
                out.append("z" if a == 1 else f"z{a}")
                total -= a
        return "".join(out)
    return re.sub(r"(?:z[0-9]*)+", run, line)


def silence_vocal(abc: str) -> tuple[str, int]:
    """Make a planned score instrumental: every note in the Vocal line becomes a rest of the same length (chord
    symbols stay), and where the Ins line of that group is silent the vocal tune moves there instead.

    Why: with empty lyrics and an "instrumental" tag the planner usually writes an all-rest Vocal line, and the
    render has no singing. But on some seeds it writes a vocal melody anyway, and then the model sings it
    (3 of 7 presets, 2026-10-17). Returns (score, number of groups changed); returns the input unchanged if the
    result would not parse."""
    lines = abc.splitlines()
    changed = 0
    i, prev_ins = 0, None
    while i < len(lines):
        if lines[i] != "V: Vocal":
            i += 1
            continue
        v = i + 1
        while v < len(lines) and lines[v].startswith(("M:", "K:")):
            v += 1
        n = v + 1
        while n < len(lines) and lines[n] != "V: Ins":
            n += 1
        ins = n + 1
        while ins < len(lines) and lines[ins].startswith(("M:", "K:")):
            ins += 1
        if v >= len(lines) or ins >= len(lines):
            break
        vocal = lines[v]
        if any(m.group(1) and m.group(1) != "z" for m in _ABC_TOKEN.finditer(vocal)):
            tune = re.sub(r'"[^"\n]*"', "", vocal)
            lines[v] = _merge_rests(_ABC_TOKEN.sub(lambda m: m.group(0) if m.group(1) is None else f"z{m.group(2)}", vocal))
            if not any(m.group(1) and m.group(1) != "z" for m in _ABC_TOKEN.finditer(lines[ins])):
                lines[ins] = re.sub(r"(?<=[A-Ga-g,'0-9])-", "", tune)  # a tie could run into a silent Ins group
                if prev_ins is not None:
                    lines[prev_ins] = re.sub(r"-(\|)$", r"\1", lines[prev_ins])  # nor may one run into this group
            changed += 1
        prev_ins = ins
        i = ins + 1
    out = "\n".join(lines) + ("\n" if abc.endswith("\n") else "")
    if changed:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
            import abc_tools
            abc_tools.parse_abc(out if out.endswith("\n") else out + "\n")
        except Exception as e:  # noqa: BLE001  (abc_tools raises AbcError; anything else is also a reason to keep the plan)
            print(f"[instrumental] could not silence the vocal line ({e}); keeping the plan", file=sys.stderr)
            return abc, 0
    return out, changed


if __name__ == "__main__":
    same = differ = 0
    for p in Path(sys.argv[1]).read_text().split():
        out, n = silence_vocal(Path(p).read_text())
        Path(p + ".py-silenced").write_text(out)
        sw = Path(p + ".swift-silenced")
        if sw.exists():
            ok = sw.read_text() == out
            same += ok; differ += not ok
            print(p, n, "identical" if ok else "DIFFERS")
        else:
            print(p, n)
    if same or differ:
        print(f"{same} identical, {differ} differ")
        sys.exit(1 if differ else 0)
