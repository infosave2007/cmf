#!/bin/bash
# The Mac's measurement queue, one run at a time: the chain at 1024 x 1024
# (clean, then with the per-class GPU profile) and at 2048 x 2048.
#   mac_queue.sh [TAGPREFIX] [what...]   what: 1024 prof 2048 base2048 edit (default: 1024 prof 2048)
#   edit needs EDIT_IMAGE (a 1024 x 1024 picture)
set -u
here=$(cd "$(dirname "$0")" && pwd)
p=${1:-mq}; shift || true
[ $# -eq 0 ] && set -- 1024 prof 2048
for what in "$@"; do
  case $what in
    1024) MIN_AREA=1024 bash "$here/mac_gen.sh" ${p}_1024 1024 1024 ;;
    prof) MIN_AREA=1024 CMF_H3_METAL_PROF=1 bash "$here/mac_gen.sh" ${p}_1024_prof 1024 1024 ;;
    2048) bash "$here/mac_gen.sh" ${p}_2048 2048 2048 ;;
    base2048) CMF_H3_METAL=0 bash "$here/mac_gen.sh" ${p}_base_2048 2048 2048 ;;
    edit) IMAGE=${EDIT_IMAGE:?a reference picture} PROMPT="$here/prompts/edit_min_cafe.txt" MIN_AREA=1024 bash "$here/mac_gen.sh" ${p}_edit_1024 1024 1024 ;;
  esac
done
echo MAC_QUEUE_DONE
