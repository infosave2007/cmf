#!/bin/bash
# upload.sh NAME LOCAL [REMOTE]: reads the HF token from stdin (never written to disk), uploads LOCAL to the private turbo repo in the background -> logs/upload_NAME.log
cd /root/rip
read -r HF_TOKEN; export HF_TOKEN
n=$1; local=$2; remote=${3:-$(basename $2)}
nohup bash -c "t0=\$(date +%s); hf upload infosave/Qwen3.8-Flash-Next-turbo-cmf $local $remote > logs/upload_$n.log 2>&1; echo \"exit \$? \$(( \$(date +%s) - t0 )) s\" >> logs/upload_$n.log; echo UPLOAD_DONE >> logs/upload_$n.log" > /dev/null 2>&1 &
echo "upload $n started"
