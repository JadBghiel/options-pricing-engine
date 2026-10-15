"""cmd line: price an option and its greeks, solve implied vol, calibrate, plot

ope price -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2
ope iv --type put -S 42 -K 40 -T 0.5 -r 0.1 --price 0.81
ope plot-greeks -K 100 -T 0.5 -r 0.05 --sigma 0.2 --out greeks.png
ope mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --paths 100000 --antithetic --control
ope plot-mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --out convergence.png
ope american --type put -S 50 -K 50 -T 0.4167 -r 0.1 --sigma 0.4
ope plot-boundary -S 50 -K 50 -T 0.4167 -r 0.1 --sigma 0.4 --out boundary.png
ope mc -S 42 -K 40 -T 0.5 -r 0.1 --sigma 0.2 --paths 10000000 --cpp --threads 8
ope asian -S 100 -K 100 -T 1 -r 0.05 --sigma 0.2 --dates 12
ope barrier -S 100 -K 100 -T 1 -r 0.05 --sigma 0.2 --barrier 120 --kind up-and-out
ope lookback -S 100 -T 1 -r 0.05 --sigma 0.2
ope heston -S 100 -K 100 -T 1 -r 0.05 --v0 0.04 --kappa 1.5 --theta 0.04 --xi 0.6 --rho -0.7
ope plot-smile -S 100 -T 1 -r 0.05 --out smile.png
ope fetch-chain --source ibkr --port 4001
ope surface tests/fixtures/spx_synthetic.csv
ope svi tests/fixtures/spx_synthetic.csv
ope calibrate tests/fixtures/spx_synthetic.csv --starts 8 --seed 0
ope plot-surface tests/fixtures/spx_synthetic.csv --out smiles.png
ope plot-term tests/fixtures/spx_synthetic.csv --out term.png

1 module per group of cmds: pricing (tiers 1 to 4), exotic (tier 5), market (tier 6), each one
adds its subparsers and returns its cmds
"""

import argparse
import sys
from collections.abc import Sequence

from ope.cli import exotic, market, pricing
from ope.cli.args import Command


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Command]]:
    parser = argparse.ArgumentParser(prog="ope", description="Options pricing engine.")
    sub = parser.add_subparsers(dest="command", required=True)
    commands: dict[str, Command] = {}
    for group in (pricing, exotic, market):
        commands |= group.register(sub)
    return parser, commands


def main(argv: Sequence[str] | None = None) -> int:
    parser, commands = build_parser()
    args = parser.parse_args(argv)
    try:
        return commands[args.command](args)
    except ValueError as e:  # bad inputs -> a 1 line msg for the user, no traceback
        print(f"error: {e}", file=sys.stderr)
        return 2
