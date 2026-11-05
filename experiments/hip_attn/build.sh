#!/usr/bin/env bash
# build the int8 flash-attention prototype shared library for gfx1100
set -euo pipefail
cd "$(dirname "$0")"
/usr/bin/hipcc --offload-arch=gfx1100 -O3 -std=c++17 -Wno-unused-result -Wno-unused-value \
    -Wno-unknown-attributes -shared -fPIC -o libattn_int8.so attn_int8.hip
echo "built: $(pwd)/libattn_int8.so"
