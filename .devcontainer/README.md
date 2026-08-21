# Devcontainer setup notes

Two configs, selectable via VS Code's "Dev Containers: Reopen in Container"
config picker:

- `devcontainer.json` (default) — headless: `cargo build/test/bench/fmt/clippy`
  only. No GPU/display access; `cargo run` will fail here.
- `gpu-amd/devcontainer.json` — adds AMD GPU passthrough (Mesa/RADV via
  `/dev/dri`) and configurable X11/Wayland display forwarding so `cargo run`
  shows the window on your host screen.

## Host prerequisites (podman)

This project uses rootless podman instead of Docker. One-time host setup:

1. Point VS Code's Dev Containers extension at podman instead of Docker —
   in VS Code settings (user or workspace), set:
   - `dev.containers.dockerPath` → `podman`
2. Ensure the rootless podman socket is running, since the Dev Containers
   extension talks to a Docker-compatible API socket:
   ```
   systemctl --user enable --now podman.socket
   ```
3. For the `gpu-amd` variant only: the container process needs `/dev/dri`
   access. Your desktop *session* likely already has this automatically via
   systemd-logind's `uaccess` udev tagging (no `video`/`render` group
   membership needed for the session itself) — but that ACL is granted to
   your logind seat, not to a podman container process, which just sees the
   device node's plain owner/group/mode (`root:render`, mode 660). So the
   container still needs your host user in the `video`/`render` groups:
   ```
   sudo usermod -aG video,render "$USER"
   ```
   (log out/in for group membership to take effect).
4. **Before opening the devcontainer in VS Code**, run any podman command
   once in a plain terminal (e.g. `podman ps`). VS Code's Electron/Chromium
   sandbox sets `NoNewPrivs=1` on its whole process tree, which blocks
   `newuidmap`/`newgidmap` from elevating their file capabilities — so
   rootless podman's usual "create a fresh user namespace" path fails with
   `newuidmap: Could not set caps` when invoked *from* VS Code. Running a
   podman command from your normal shell first creates podman's long-lived
   rootless "pause" process (`catatonit`); VS Code's sandboxed process can
   then join that existing namespace instead of creating a new one, which
   doesn't require the blocked capability elevation. This pause process
   doesn't survive logout/reboot, so repeat this step after those.

## Display forwarding (gpu-amd variant)

The `gpu-amd` config mounts `/tmp/.X11-unix` and your `$XDG_RUNTIME_DIR`
unconditionally, and forwards whichever of `DISPLAY`/`WAYLAND_DISPLAY` is set
in your host shell at container-create time. No separate X11-only/Wayland-only
config is needed — it follows whatever your host session actually uses.

## Build caches

`~/.cargo/registry` and `~/.cargo/git` are shared, global named volumes
across all workspaces and configs. The `target/` directory is a named volume
scoped per workspace folder (`hypercube-target-<folder-basename>`), so
multiple jj workspaces or git worktrees checked out to differently-named
directories each get their own build cache and can build in parallel without
blocking on Cargo's build lock. Two worktrees that happen to share the same
directory basename will still share a `target` volume — rename one if you
need them fully independent.
