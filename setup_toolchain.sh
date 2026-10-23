#!/usr/bin/env bash
# Installs a user-local CUDA 13 compiler/runtime (pip wheels, no root needed) and
# fetches the SDL2 / OpenGL headers into third_party/include without root.
# The runtime libraries (libSDL2-2.0.so.0, libGL.so.1, libz) must be present on the system.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${CUDA_VENV:-$HOME/.local/cuda13-venv}"
if [ ! -x "$VENV/bin/pip" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-crt nvidia-cuda-cccl
CUDA_HOME=$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/cu13 | head -1)
echo "CUDA toolkit: $CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -1

if [ ! -f third_party/include/SDL2/SDL.h ] || [ ! -f third_party/include/GL/gl.h ]; then
  tmp=$(mktemp -d)
  ( cd "$tmp" && apt-get download libsdl2-dev libgl-dev && for d in *.deb; do dpkg -x "$d" x; done )
  mkdir -p third_party/include
  cp -r "$tmp"/x/usr/include/SDL2 "$tmp"/x/usr/include/GL "$tmp"/x/usr/include/KHR third_party/include/
  cp "$tmp"/x/usr/include/*-linux-gnu/SDL2/* third_party/include/SDL2/
  rm -rf "$tmp"
fi
echo "Headers: third_party/include"
echo
echo "Build with:  make CUDA_HOME=$CUDA_HOME"
