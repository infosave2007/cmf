#!/bin/bash
# Fetch the raw text sources for the Hermes core corpus into /root/corpus/raw and report sizes.
# Each source is licensed for reuse with attribution or share-alike; LICENSES.tsv is the record.
# Everything resumes: a rerun skips what is present.
R=/root/corpus/raw; mkdir -p $R; cd $R || exit 1
L=/root/corpus/fetch.log; : > $L
say() { echo "$(date +%H:%M:%S) $*" >> $L; }

clone() { # dir url
  [ -d "$1/.git" ] || git clone -q --depth 1 "$2" "$1" 2>> $L
  say "clone $1: $(du -sh "$1" 2>/dev/null | cut -f1)"
}

clone tldr https://github.com/tldr-pages/tldr
clone powershell-docs https://github.com/MicrosoftDocs/PowerShell-Docs
clone windows-powershell-docs https://github.com/MicrosoftDocs/windows-powershell-docs
clone freebsd-doc https://github.com/freebsd/freebsd-doc
clone fedora-quick-docs https://pagure.io/fedora-docs/quick-docs.git
clone debian-handbook https://salsa.debian.org/hertzog/debian-handbook.git
clone git-docs https://github.com/git/git

# Stack Exchange dumps, CC BY-SA: the sites that talk about operating systems and commands.
mkdir -p se; cd se || exit 1
for s in unix.stackexchange.com askubuntu.com serverfault.com superuser.com; do
  f=$s.7z
  curl -sSL -C - --retry 10 --retry-delay 5 -o $f "https://archive.org/download/stackexchange/$f" 2>> $L
  say "se $s: $(du -sh $f 2>/dev/null | cut -f1)"
done
cd $R || exit 1

# Man pages of a wide Debian/Ubuntu command set, as text.
apt-get install -y -q man-db manpages manpages-dev manpages-posix groff-base p7zip-full >> $L 2>&1
mkdir -p man; cd man || exit 1
unminimize_note=/etc/dpkg/dpkg.cfg.d/excludes
[ -f $unminimize_note ] && sed -i 's#^path-exclude=/usr/share/man#\#&#' $unminimize_note
say "man: pages available $(find /usr/share/man -type f | wc -l)"
cd $R || exit 1

say "sizes:"; du -sh $R/* >> $L 2>&1
echo CORPUS-FETCH-DONE >> $L
