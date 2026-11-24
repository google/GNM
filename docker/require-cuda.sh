#!/bin/sh
set -eu
python -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable; refusing to start the CUDA service")'
exec "$@"
