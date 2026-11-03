#!/bin/bash
# Second batch: more operating system and devops text, Russian Q&A, and the tagged part of Stack Overflow.
R=/root/corpus/raw; L=/root/corpus/fetch2.log; : > $L; cd $R || exit 1
say() { echo "$(date +%H:%M:%S) $*" >> $L; }
clone() { [ -d "$1/.git" ] || git clone -q --depth 1 "$2" "$1" 2>> $L; say "clone $1: $(du -sh "$1" 2>/dev/null | cut -f1)"; }
clone docker-docs https://github.com/docker/docs
clone kubernetes https://github.com/kubernetes/website
clone cpython https://github.com/python/cpython
mkdir -p arch-wiki; cd arch-wiki || exit 1
f=$(curl -sS -m 30 https://geo.mirror.pkgbuild.com/extra/os/x86_64/ | grep -o 'arch-wiki-docs-[^"]*zst' | head -1)
curl -sSL -C - --retry 10 -o "$f" "https://geo.mirror.pkgbuild.com/extra/os/x86_64/$f" 2>> $L && tar --zstd -xf "$f" 2>> $L
say "arch wiki: $(find . -name '*.html' | wc -l) pages"; cd $R || exit 1
cd se || exit 1
for f in ru.stackoverflow.com.7z stackoverflow.com-Posts.7z; do
  # archive.org holds a single connection to about 1.6 MB/s
  aria2c -q -c -x 16 -s 16 --retry-wait=5 --max-tries=20 -d . -o $f "https://archive.org/download/stackexchange/$f" 2>> $L
  say "se $f: $(du -sh $f | cut -f1)"
done
J=/root/corpus/jsonl; S=/root/cmf/scripts/hermes-core/corpus_se.py
7z x -so ru.stackoverflow.com.7z Posts.xml 2>> $L | python3 $S ru.stackoverflow.com - $J/ru.stackoverflow.com.jsonl >> $L 2>&1
7z x -so stackoverflow.com-Posts.7z Posts.xml 2>> $L | python3 $S stackoverflow.com - $J/stackoverflow.com.jsonl --min-score 2 \
  --tags bash,shell,sh,zsh,linux,unix,ubuntu,debian,centos,fedora,alpine,macos,osx,homebrew,windows,powershell,cmd,batch-file,wsl,git,github-actions,ssh,scp,rsync,curl,wget,sed,awk,grep,find,xargs,cron,crontab,systemd,nginx,apache,docker,docker-compose,dockerfile,kubernetes,kubectl,ansible,terraform,makefile,tar,chmod,permissions,sudo,environment-variables,pip,virtualenv,conda,apt,yum,dnf,iptables,firewall,tmux,vim >> $L 2>&1
du -sh $J/* >> $L
echo FETCH2-DONE >> $L
