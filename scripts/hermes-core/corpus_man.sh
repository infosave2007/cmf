#!/bin/bash
# Man pages of a wide command set as plain text. The image strips /usr/share/man, so the packages
# are downloaded and only their man pages are unpacked; nothing here gets installed.
R=/root/corpus/man; O=/root/corpus/man.log; : > $O
mkdir -p $R/deb $R/x $R/txt; cd $R/deb || exit 1
PK="coreutils util-linux procps iproute2 net-tools iputils-ping iputils-tracepath openssh-client openssh-server
 systemd systemd-timesyncd git curl wget tar gzip bzip2 xz-utils zstd unzip zip findutils grep sed gawk mawk bash zsh dash fish
 vim nano less rsync cron sudo mount e2fsprogs xfsprogs btrfs-progs dosfstools parted fdisk lvm2 lsof strace ltrace tcpdump
 iptables nftables ufw nmap netcat-openbsd socat dnsutils bind9-host traceroute mtr-tiny whois ethtool
 nginx apache2 postgresql-client mariadb-client redis-tools sqlite3 docker.io podman
 python3 python3-pip python3-venv nodejs npm gcc g++ make cmake gdb binutils pkg-config autoconf automake
 apt dpkg debconf snapd flatpak man-db manpages manpages-dev manpages-posix file tree htop tmux screen jq bc dc
 ca-certificates openssl gnupg acl attr passwd adduser shadow-utils hostname tzdata locales kmod pciutils usbutils lshw dmidecode
 sysstat iotop nethogs logrotate rsyslog ansible patch diffutils ed cpio pv moreutils parallel time watch"
for p in $PK; do ls "$p"_*.deb >/dev/null 2>&1 || apt-get download -q "$p" >> $O 2>&1; done
n=0
for d in *.deb; do
  rm -rf "$R/x/cur"; mkdir -p "$R/x/cur"
  dpkg-deb -x "$d" "$R/x/cur" 2>> $O || continue
  find "$R/x/cur/usr/share/man" -type f -path '*/share/man/man[1-8]/*' 2>/dev/null | while read -r f; do
    b=$(basename "$f" .gz); pkg=${d%%_*}
    case "$f" in *.gz) zcat "$f" ;; *) cat "$f" ;; esac | MANWIDTH=100 groff -t -e -P-c -Tutf8 -man 2>/dev/null | col -bx > "$R/txt/$pkg--$b.txt" 2>/dev/null
  done
  n=$((n+1))
done
rm -rf "$R/x"
echo "packages $n, pages $(ls $R/txt | wc -l), size $(du -sh $R/txt | cut -f1)" >> $O
echo MAN-DONE >> $O
