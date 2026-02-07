#!/bin/bash
# libgomp.so.1 찾아서 LD_LIBRARY_PATH에 추가
LIB_PATH=$(find / -name 'libgomp.so.1' 2>/dev/null | head -1)
if [ -n "$LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$(dirname "$LIB_PATH"):$LD_LIBRARY_PATH
    echo "[OK] Found libgomp at: $LIB_PATH"
else
    echo "[WARN] libgomp.so.1 not found, trying to locate gcc libs..."
    GCC_LIB=$(find /nix -path '*/lib/libgomp*' 2>/dev/null | head -1)
    if [ -n "$GCC_LIB" ]; then
        export LD_LIBRARY_PATH=$(dirname "$GCC_LIB"):$LD_LIBRARY_PATH
        echo "[OK] Found gcc lib at: $GCC_LIB"
    fi
fi
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
exec gunicorn app:app --bind 0.0.0.0:8080 --timeout 120 --workers 1
