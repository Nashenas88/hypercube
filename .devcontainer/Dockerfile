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
RUN cargo install --root /usr/local jj-cli r3bl-rust-analyzer-mcp-server --locked

# Node 22 or newer is required by the Claude Code devcontainer feature, which
# installs Node itself only when absent; Debian packages 18.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash dev

# Package install scripts are arbitrary code; dev's npm does not run them.
RUN printf 'ignore-scripts=true\n' > /home/dev/.npmrc \
    && chown dev:dev /home/dev/.npmrc

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
