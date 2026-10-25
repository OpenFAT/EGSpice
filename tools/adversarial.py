"""Adversarial sizes: decks no human writes, at depths and counts that break
recursion and quadratic paths. Oracle: espice exits 0 or refuses with a
diagnostic (exit 1-127); a signal, a timeout or an empty diagnostic fails.

    python3 tools/adversarial.py zig-out/bin/espice
"""
import subprocess, sys, tempfile
from pathlib import Path

N = 100_000
DECKS = {
    "nested_subckt": "".join(f".subckt s{i} a b\nx{i} a b s{i + 1}\n.ends\n" for i in range(2000))
    + ".subckt s2000 a b\nr1 a b 1k\n.ends\nv1 a 0 1\nx0 a 0 s0\n.op\n",
    "nested_parens": f"v1 a 0 {{{'(' * N}1{')' * N}}}\nr1 a 0 1k\n.op\n",
    "long_continuation": "r1 a 0\n" + "+ 1k\n" * N + "v1 a 0 1\n.op\n",
    "many_params": "".join(f".param p{i}=p{i - 1}+1\n" for i in range(1, N)) + ".param p0=0\nv1 a 0 1\nr1 a 0 {p1}\n.op\n",
    "huge_ladder": "v1 n0 0 1\n" + "".join(f"r{i} n{i} n{i + 1} 1\nc{i} n{i + 1} 0 1p\n" for i in range(N)) + ".op\n",
    "long_name": f"v1 {'a' * N} 0 1\nr1 {'a' * N} 0 1k\n.op\n",
}

exe = Path(sys.argv[1]).resolve()
bad = 0
with tempfile.TemporaryDirectory() as work:
    for name, body in DECKS.items():
        deck = Path(work) / f"{name}.sp"
        deck.write_text(f"{name}\n{body}.end\n")
        try:
            r = subprocess.run(["bash", "-c", 'ulimit -v 8000000; exec "$@"', "_", str(exe), "--backend", "cpu", "-b", "-r", f"{work}/out.raw", str(deck)],
                               capture_output=True, text=True, timeout=300)
            ok = r.returncode == 0 or (0 < r.returncode < 128 and r.stderr.strip())
            verdict = f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            ok, verdict = False, "timeout"
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {verdict}")
print(f"adversarial: {len(DECKS) - bad} of {len(DECKS)} handled")
sys.exit(1 if bad else 0)
