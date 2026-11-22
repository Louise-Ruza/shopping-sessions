# Shopping Sessions: native Rust pipeline.
# The scorer builds this natively on linux/amd64 (Intel Xeon, Cascade Lake or
# newer). No FROM line pins a platform, so it also builds natively on arm64.

FROM rust:slim-bookworm AS build
ARG TARGETARCH
ARG ENABLE_PGO=1
# PGO_NONCE only busts the build cache so scripts/pgo_export.sh can draw independent profiles.
ARG PGO_NONCE=0
WORKDIR /src
# Install llvm-tools in its own cached layer before copying src so rebuilds don't
# re-fetch it and any transient download hiccup is retried loudly (never silent).
RUN set -eu; \
    if [ "$ENABLE_PGO" = "1" ]; then \
        rustup component add llvm-tools-preview || rustup component add llvm-tools-preview || echo "WARN: llvm-tools-preview unavailable, will fall back to non-PGO"; \
    fi
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# x86-64-v3 (AVX2/BMI1/BMI2/LZCNT/POPCNT) is available on every Haswell-or-newer
# Xeon, including all scorer CPUs. Plain glibc + no JCC branch padding was verified
# -90 ms faster on the live scorer than libjemalloc2 + 32B branch alignment.
# Build-time PGO trains on a self-contained synthetic dataset (zero external
# downloads) and falls back cleanly if llvm-profdata is unavailable offline.
# Training (src/pgo/train.sh) is single-threaded with atomic counters, so the same
# source + rustc always yields the same binary (NCPU=4 training varied +-25 ms at 60M).
RUN set -eu; \
    BASE_FLAGS=""; \
    if [ "$TARGETARCH" = "amd64" ] || [ "$(uname -m)" = "x86_64" ]; then \
        BASE_FLAGS="-C target-cpu=x86-64-v3"; \
    fi; \
    HOST="$(rustc -vV | sed -n 's/^host: //p')"; \
    PROFDATA="$(rustc --print sysroot)/lib/rustlib/$HOST/bin/llvm-profdata"; \
    if [ "$ENABLE_PGO" = "1" ] && [ -x "$PROFDATA" ]; then \
        echo "=== PGO: ENABLED (host=$HOST rustc=$(rustc --version)) ==="; \
        WORK="$(mktemp -d)"; \
        sh src/pgo/train.sh "$WORK" "$PROFDATA" "$HOST" "$BASE_FLAGS"; \
        RUSTFLAGS="$BASE_FLAGS -Cprofile-use=$WORK/merged.profdata" \
            cargo build --release --locked --target "$HOST"; \
        strip -o /shopping-sessions "target/$HOST/release/shopping-sessions"; \
        mkdir -p /pgo-out; cp "$WORK/merged.profdata" /pgo-out/; \
        rm -rf "$WORK"; \
    else \
        echo "=== PGO: FALLBACK (ENABLE_PGO=$ENABLE_PGO profdata=$PROFDATA) ==="; \
        RUSTFLAGS="$BASE_FLAGS" cargo build --release --locked; \
        strip -o /shopping-sessions target/release/shopping-sessions; \
    fi

# Preserve the trained v3 artifact and profile in the stock build layer.
FROM build AS wide
COPY dispatch.rs /src/dispatch.rs
RUN set -eu; \
    mv /shopping-sessions /shopping-sessions-v3; \
    HOST="$(rustc -vV | sed -n 's/^host: //p')"; \
    case "$HOST" in \
        x86_64-*) \
            FLAGS="-C target-cpu=x86-64-v4"; \
            if [ -f /pgo-out/merged.profdata ]; then FLAGS="$FLAGS -Cprofile-use=/pgo-out/merged.profdata"; fi; \
            RUSTFLAGS="$FLAGS" cargo build --release --locked --target "$HOST"; \
            strip -o /shopping-sessions "target/$HOST/release/shopping-sessions";; \
        *) cp /shopping-sessions-v3 /shopping-sessions;; \
    esac; \
    rustc --edition=2021 -O -Cstrip=symbols dispatch.rs -o /dispatch

# Export unstripped symbols from the same scoring build to the research profiler.
FROM wide AS research-symbol-copy
RUN set -eu; host="$(rustc -vV | sed -n 's/^host: //p')"; \
    for exe in "target/$host/release/shopping-sessions" target/release/shopping-sessions; do \
        if [ -f "$exe" ]; then cp "$exe" /research-symbols; exit 0; fi; \
    done; exit 1
FROM scratch AS profile-symbols
COPY --from=research-symbol-copy /research-symbols /shopping-sessions

# Minimal runtime: plain glibc + libgcc only, no shell. Matches bookworm's glibc.
FROM gcr.io/distroless/cc-debian12
COPY --from=wide /shopping-sessions /shopping-sessions
COPY --from=wide /shopping-sessions-v3 /shopping-sessions-v3
COPY --from=wide /dispatch /dispatch
# Scorer env: DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch NCPU=8 MEM_GB=24.
# The binary writes only under OUTPUT_DIR and needs no writable root or /tmp.
ENV DATA_DIR=/data OUTPUT_DIR=/output
USER 1000:1000
ENTRYPOINT ["/dispatch"]
