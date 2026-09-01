FROM rust:slim AS base

# winit/iced link against these at build time on Linux regardless of which
# display backend (or none) is available at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    git \
    libx11-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

# Installed outside CARGO_HOME, which a named volume mounts over at runtime and
# would otherwise mask.
RUN cargo install --root /usr/local jj-cli --locked

RUN useradd --create-home --shell /bin/bash dev

# Workspaces are bind-mounted under /workspaces, which a secondary jj workspace
# also writes to, to place a symlink beside itself for its main repo.
RUN mkdir -p /workspaces /home/dev/jj-main \
    && chown dev:dev /workspaces /home/dev/jj-main

USER dev
WORKDIR /workspaces

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
