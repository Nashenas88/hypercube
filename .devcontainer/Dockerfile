FROM rust:slim

# winit/iced link against these at build time on Linux regardless of which
# display backend (or none) is available at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libx11-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash dev
USER dev
WORKDIR /workspace

COPY --chown=dev:dev rust-toolchain.toml /tmp/rust-toolchain.toml
RUN cd /tmp && rustup show
