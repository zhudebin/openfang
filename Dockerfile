# syntax=docker/dockerfile:1
FROM rust:1-slim-bookworm AS builder
WORKDIR /build

# 配置国内镜像源（Debian apt + Cargo crates.io）
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && mkdir -p /usr/local/cargo \
    && printf '[source.crates-io]\nreplace-with = "ustc"\n\n[source.ustc]\nregistry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"\n' > /usr/local/cargo/config.toml

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY xtask ./xtask
COPY agents ./agents
COPY packages ./packages
RUN cargo build --release --bin openfang

FROM ubuntu:24.04
# 配置国内镜像源（Ubuntu apt）
RUN sed -i 's|archive.ubuntu.com|mirrors.ustc.edu.cn|g; s|security.ubuntu.com|mirrors.ustc.edu.cn|g; s|ports.ubuntu.com|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/openfang /usr/local/bin/
COPY --from=builder /build/agents /opt/openfang/agents
EXPOSE 4200
VOLUME /data
ENV OPENFANG_HOME=/data
ENTRYPOINT ["openfang"]
CMD ["start"]
