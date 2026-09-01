# Devcontainer setup notes

Two configs, selectable via VS Code's "Dev Containers: Reopen in Container"
config picker:

- `devcontainer.json` (default) — headless: `cargo build/test/bench/fmt/clippy`
  only. No GPU/display access; `cargo run` will fail here.
- `gpu-amd/devcontainer.json` — adds AMD GPU passthrough (Mesa/RADV via
  `/dev/dri`) and configurable X11/Wayland display forwarding so `cargo run`
  shows the window on your host screen.

Both build from the single `Dockerfile`, which has a shared `base` stage and a
`gpu` stage layered on top of it; each config picks its stage with
`build.target`. The GPU image therefore reuses the base image's layers rather
than repeating its apt and rustup steps.

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

## jj and git

`git` comes from apt. `jj` is built with `cargo install --root /usr/local`,
which puts it in `/usr/local/bin` rather than under `CARGO_HOME`, where the
cargo cache volume would mask it.

`JJ_CONFIG` names two files, and the later one wins:

1. `~/.config/jj/config.toml`, bind-mounted read-only from the host, which
   supplies `user.name` and `user.email` so no identity is committed here. This
   file must exist on the host, or the container will not start.
2. `jj-config.toml`, which pins the pager and diff formatter to jj's built-ins.
   The host config selects `delta`; without the override `jj diff` fails with
   `Error executing 'delta'` inside the container.

No revset aliases are set: jj's built-in `trunk()` already resolves to
`main@origin` here.

The repo-level config `.jj/repo/config.toml` is a symlink to a path outside the
workspace, so it dangles in the container. jj treats that as a missing per-repo
config, warns once, and generates an empty one under its own config directory —
the host's `.jj` is not modified.

## jj workspaces

Workspaces are mounted at `/workspaces/<folder-basename>` rather than a fixed
path, so a secondary jj workspace and its main repo can sit side by side.

A secondary workspace's `.jj/repo` is a file holding the path of the main repo's
`.jj/repo`, relative to `.jj` — a location that does not exist in the container,
and one that must be writable, since it holds the operation log and the commit
store. Two hooks bridge that without rewriting the host's `.jj`:

- `initializeCommand` runs `jj-main-link.sh` on the host, pointing the gitignored
  `.devcontainer/.jj-main` symlink at the main repo (the checkout itself, for a
  main workspace). It gives the config a fixed path to bind-mount, whatever the
  main repo is named or wherever it lives, and mounts only that one repository.
- `postCreateCommand` runs `jj-workspace-link.sh` in the container, which
  recreates the location the pointer resolves to as a symlink to that mount.

A main workspace needs neither: `.jj/repo` is a directory, so the second script
exits immediately and the mount is just the workspace itself.

## File ownership on bind mounts

Both configs pass `--userns=keep-id:uid=1000,gid=1000`. Rootless podman
otherwise maps the container user to a subordinate uid, so bind-mounted host
files show up owned by `root` and the `dev` user cannot write to them — builds
still work, because `target/` is a named volume, but editing tracked files or
committing from inside the container fails. `keep-id` maps the host uid and gid
straight through, which lines up with `dev` being uid 1000 in the image.

## Display forwarding (gpu-amd variant)

The `gpu-amd` config mounts `/tmp/.X11-unix` and your `$XDG_RUNTIME_DIR`
unconditionally, and forwards whichever of `DISPLAY`/`WAYLAND_DISPLAY` is set
in your host shell at container-create time. No separate X11-only/Wayland-only
config is needed — it follows whatever your host session actually uses.

## Build caches

`hypercube-cargo-cache` is a shared, global named volume across all workspaces
and configs, mounted over `CARGO_HOME` — which the `rust` base image sets to
`/usr/local/cargo`, not `~/.cargo`. Mounting the whole cargo home rather than
its `registry`/`git` subdirectories means the volume seeds from a directory that
already exists in the image, so it inherits that directory's permissive mode and
keeps the cargo and rustup shims reachable.

One consequence: the volume masks later image changes under `/usr/local/cargo`,
since a named volume is only seeded when it is first created. Tools that must
track the image are installed outside it.

The `target/` directory is a named volume scoped per workspace folder
(`hypercube-target-<folder-basename>`), so multiple jj workspaces or git
worktrees checked out to differently-named directories each get their own build
cache and can build in parallel without blocking on Cargo's build lock. Two
worktrees that happen to share the same directory basename will still share a
`target` volume — rename one if you need them fully independent.
