#!/bin/sh
# Points .devcontainer/.jj-main at the repository holding this checkout's jj
# state, so the devcontainer has a fixed path to bind-mount.
#
# For the main workspace that is the checkout itself. For a secondary jj
# workspace, .jj/repo is a file naming the main repo's .jj/repo directory,
# relative to .jj, and the link resolves to that repo instead.
#
# Runs on the host, before the container is created.
set -eu

workspace=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ -f "$workspace/.jj/repo" ]; then
    pointer=$(cat "$workspace/.jj/repo")
    target=$(CDPATH= cd -- "$workspace/.jj/$(dirname -- "$pointer")/.." && pwd)
else
    target=$workspace
fi

ln -sfn "$target" "$workspace/.devcontainer/.jj-main"
