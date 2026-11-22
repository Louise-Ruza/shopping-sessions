#!/bin/sh
# Build-time PGO training, shared by the Dockerfile and scripts/pgo_export.sh.
# Usage: sh src/pgo/train.sh <WORK_DIR> <PROFDATA_BIN> <HOST_TRIPLE> "<BASE_RUSTFLAGS>"
# Writes <WORK_DIR>/merged.profdata. Run from the crate root.
# NCPU=1 (no LLVM-version-specific flags): NCPU=4 training (racy counters + work-stealing splits) made every
# rebuild of identical source a different binary (+-25 ms at 60M). NCPU=1 is deterministic.
set -eu
WORK="$1"; PROFDATA="$2"; HOST="$3"; BASE_FLAGS="$4"
rustc -O src/pgo_gen.rs -o "$WORK/pgo_gen"
"$WORK/pgo_gen" "$WORK/data1" 1729
"$WORK/pgo_gen" "$WORK/data2" 4242
RUSTFLAGS="$BASE_FLAGS -Cprofile-generate=$WORK/raw" \
    cargo build --release --locked --target "$HOST" --target-dir "$WORK/inst"
PGO_TRAIN=1 NCPU=1 "$WORK/inst/$HOST/release/shopping-sessions" "$WORK/data1" "$WORK/out1"
PGO_TRAIN=1 NCPU=1 FORCE_SHARD_BITS=10 "$WORK/inst/$HOST/release/shopping-sessions" "$WORK/data2" "$WORK/out2"
"$PROFDATA" merge -o "$WORK/merged.profdata" "$WORK"/raw/*.profraw
