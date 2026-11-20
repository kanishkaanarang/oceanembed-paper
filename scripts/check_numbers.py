#!/usr/bin/env python
"""List numbers in .tex files that are not in docs/results_ledger.md or the generated table.
OK = equal in value; WEAK = matches only by rounding a more precise ledger number (review by hand);
UNMATCHED = not found. Ignores comments, years, the 15 depth values, integers <= 31, commit hashes.
Usage: python scripts/check_numbers.py [file.tex ...]   (exit 1 if anything is UNMATCHED)"""
import re, sys
from decimal import Decimal, ROUND_HALF_UP

LEDGER = "docs/results_ledger.md"
TABLE = "paper/tables/tab_main.tex"
DEFAULT = ["paper/sections/results_v2.tex"]
YEARS = set(range(2010, 2031))
DEPTHS = {0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000}
SMALL = 31
NUM = re.compile(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])')
STRIP = re.compile(r'\\(?:label|ref|cite|citep|citet|input|includegraphics|cline|multicolumn|begin|end|'
                   r'section|subsection|usepackage|bibliography|bibliographystyle)(?:\[[^\]]*\])?(?:\{[^}]*\})+')

def clean(line):
    line = re.sub(r'(?<!\\)%.*$', '', line)
    line = STRIP.sub(' ', line)
    return line.replace('{,}', ',')

def tokens(text):
    return [(m.start(), m.end(), m.group().replace(',', '')) for m in NUM.finditer(text)]

def load_source(path, is_tex):
    out = []
    for line in open(path, encoding="utf-8"):
        t = clean(line) if is_tex else re.sub(r'\(not [^)]*\)', ' ', line)
        out += [Decimal(tok) for _, _, tok in tokens(t)]
    return out

src = load_source(LEDGER, False) + load_source(TABLE, True)
strong = set(src)

def classify(tok, pct=False):
    if '.' not in tok:
        v = int(tok)
        if v in YEARS or v in DEPTHS or v <= SMALL:
            return "IGNORED"
    d = Decimal(tok)
    if d in strong:
        return "OK"
    p = -d.as_tuple().exponent
    q = Decimal(1).scaleb(-p)
    for x in src:
        if -x.as_tuple().exponent > p and x.quantize(q, ROUND_HALF_UP) == d:
            return "WEAK"
    if pct:
        for x in src:
            if (x * 100).quantize(q, ROUND_HALF_UP) == d:
                return "PCT"
    return "UNMATCHED"

files = sys.argv[1:] or DEFAULT
bad = 0
for f in files:
    counts = {"OK": 0, "WEAK": 0, "PCT": 0, "UNMATCHED": 0, "IGNORED": 0}
    print(f"=== {f} ===")
    for i, raw in enumerate(open(f, encoding="utf-8"), 1):
        line = clean(raw)
        for s, e, tok in tokens(line):
            c = classify(tok, line[e:e+2] == "\\%"); counts[c] += 1
            if c in ("WEAK", "UNMATCHED", "PCT"):
                print(f"{c:9s} line {i}: {tok:>8}   ...{line[max(0, s-35):e+15].strip()}...")
    print("summary:", counts)
    bad += counts["UNMATCHED"]
sys.exit(1 if bad else 0)
