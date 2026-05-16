#!/usr/bin/env bash
# Compile the agent crate in release mode.
set -e
cd "$(dirname "$0")"
cargo build --release
echo "[build] starters/rust/target/release/agent built" >&2
