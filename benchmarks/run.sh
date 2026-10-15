#!/usr/bin/env bash
# every benchmark number in the rdme and REPORT.md, in 1 cmd: bash benchmarks/run.sh
# c++ side w/ google benchmark, python side w/ pytest-benchmark, then benchmarks/RESULTS.md
# close other heavy programs first, the thread scaling needs idle cores
set -euo pipefail
cd "$(dirname "$0")/.."

UV="${UV:-uv}"  # UV=~/.local/bin/uv bash benchmarks/run.sh if uv isnt on the PATH
OUT=build/bench
mkdir -p "$OUT"

# release on purpose, a debug build is 5 to 10x slower and tells nothing
cmake -S . -B build/bench-cpp -DCMAKE_BUILD_TYPE=Release -DOPE_BUILD_BENCHMARKS=ON
cmake --build build/bench-cpp --target ope_bench -j
build/bench-cpp/ope_bench \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out="$OUT/cpp.json" \
  --benchmark_out_format=json

"$UV" sync --locked --all-extras
"$UV" run pytest benchmarks --no-cov -q --benchmark-json="$OUT/python.json" --benchmark-columns=median,iqr,rounds

"$UV" run python benchmarks/report.py "$OUT"
