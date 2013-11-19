#!/bin/bash
if [ -f ~/ni_env/bin/activate ]; then
    source ~/ni_env/bin/activate
fi

echo "$(dirname "$1")"

cd "$(dirname "$1")"


ipython $@
