#!/bin/sh
# Makes a secondary jj workspace's .jj/repo pointer resolve inside the
# container.
#
# The pointer is a path relative to .jj, so it names a location beside the
# workspace that does not exist in the container. Recreating it as a symlink to
# the bind-mounted main repo lets jj follow the pointer unchanged, leaving the
# host's .jj untouched.
#
# Runs in the container, after it is created. A main workspace has .jj/repo as a
# directory and needs nothing.
set -eu

main_mount=/home/dev/jj-main
workspace=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

[ -f "$workspace/.jj/repo" ] || exit 0

pointer=$(cat "$workspace/.jj/repo")
main_path=$(realpath -m "$workspace/.jj/$(dirname -- "$pointer")/..")

if [ -e "$main_path" ]; then
    exit 0
fi

mkdir -p "$(dirname -- "$main_path")"
ln -sfn "$main_mount" "$main_path"
