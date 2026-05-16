#!/usr/bin/env bash
# Compile main.cpp + agent.hpp into a single binary called `agent`.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o agent main.cpp
echo "[build] starters/cpp/agent built" >&2
