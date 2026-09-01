FROM rust:slim AS base

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

# Vulkan drivers and fonts, for rendering against a GPU and display shared in
# from the host.
FROM base AS gpu

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-vulkan-drivers \
    libvulkan1 \
    vulkan-tools \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
USER dev
