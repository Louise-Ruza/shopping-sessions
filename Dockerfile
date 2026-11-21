# Fixed selected solution; portable AMD64 baseline and runtime SIMD dispatch.
FROM rust:1.90-slim-bookworm AS build
ENV RUSTFLAGS="-C target-cpu=x86-64"
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo test --locked && cargo build --release --locked \
    && strip -o /shopping-sessions target/release/shopping-sessions

FROM gcr.io/distroless/cc-debian12
COPY --from=build /shopping-sessions /shopping-sessions
COPY ["LICENSE", "/licenses/LICENSE"]
COPY third_party /licenses/third_party
ENV DATA_DIR=/data OUTPUT_DIR=/output
USER 1000:1000
ENTRYPOINT ["/shopping-sessions"]
