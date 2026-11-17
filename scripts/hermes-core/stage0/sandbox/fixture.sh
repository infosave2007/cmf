#!/bin/bash
# Deterministic fixture of the R sandbox: /data, read-only for the task user.
set -eu
mkdir -p /data/logs /data/src /data/big /data/empty /data/docs/sub /data/site/css /data/site/img

awk 'BEGIN{ split("INFO INFO WARN INFO ERROR INFO DEBUG INFO INFO WARN",lv," "); split("auth db cache api worker",sv," ");
  for(i=1;i<=200;i++){ printf "2026-03-%02d %02d:%02d:%02d %s %s: request %d took %d ms\n",
    (i%28)+1,(i*5)%24,(i*13)%60,(i*29)%60, lv[(i*7)%10+1], sv[(i*3)%5+1], 1000+i, (i*37)%900+10 } }' > /data/app.log

awk 'BEGIN{ split("/index.html /about /api/users /api/orders /login /static/app.js /img/logo.png",p," "); split("200 200 200 404 200 500 301 200 403 200",st," ");
  for(i=1;i<=150;i++){ printf "10.0.%d.%d - - [12/Mar/2026:%02d:%02d:%02d +0000] \"GET %s HTTP/1.1\" %s %d\n",
    (i*3)%4,(i*17)%50+1,(i*7)%24,(i*11)%60,(i*23)%60, p[(i*5)%7+1], st[(i*9)%10+1], (i*131)%5000+120 } }' > /data/access.log

{ echo "id,name,city,age"; awk 'BEGIN{ split("Anna Boris Clara David Elena Felix Greta Hugo Irina Jonas Karl Lena Marta Nils Olga Pavel Quinn Rosa Sven Tara",n," "); split("Berlin Madrid Oslo Rome Vienna",c," ");
  for(i=1;i<=40;i++){ printf "%d,%s,%s,%d\n", i, n[(i*7)%20+1], c[(i*3)%5+1], 20+(i*11)%45 } }'; } > /data/users.csv

awk 'BEGIN{ split("pear apple fig kiwi plum apple lime pear date fig mango kiwi apple peach lime",n," ");
  for(i=1;i<=60;i++) print n[(i*4)%15+1] }' > /data/names.txt

awk 'BEGIN{ for(i=1;i<=50;i++) print (i*7919)%1000 }' > /data/numbers.txt

cat > /data/notes.txt <<'TXT'
Meeting notes for the storage migration.
TODO: order two new disks for the backup server.
The old array holds 12 volumes and 3 of them are read only.
todo: ask Clara about the retention policy.
Budget was approved on Monday.
Network team moves the switch on Friday.
TODO: update the runbook after the move.
The test restore took 40 minutes for 200 GB.
Nobody owns the legacy share anymore.
FIXME: the cron entry still points to the old host.
Final check is planned for the end of the month.
TXT

cat > /data/config.ini <<'TXT'
[server]
host = 0.0.0.0
port = 8080
workers = 4
debug = false

[database]
host = db.internal
port = 5432
name = shop
timeout = 30
TXT

for i in 1 2 3 4 5; do
  n=$(( (i*i*37) % 230 + 5 ))
  awk -v n="$n" -v f="$i" 'BEGIN{ for(i=1;i<=n;i++) printf "line %d of file %d status=%s\n", i, f, (i%9==0)?"fail":"ok" }' > /data/logs/app.$i.log
done
awk 'BEGIN{ for(i=1;i<=80;i++) printf "archived entry %d\n", i }' | gzip -n > /data/logs/old.log.gz

cat > /data/src/main.py <<'TXT'
import sys
import json
from util import load, total

def main(path):
    # TODO: validate the path
    rows = load(path)
    print(total(rows))

if __name__ == "__main__":
    main(sys.argv[1])
TXT
cat > /data/src/util.py <<'TXT'
import json

def load(path):
    with open(path) as f:
        return json.load(f)

def total(rows):
    # TODO: skip negative values
    return sum(r["value"] for r in rows)

def mean(rows):
    return total(rows) / len(rows)
TXT
cat > /data/src/test_util.py <<'TXT'
from util import total, mean

def test_total():
    assert total([{"value": 2}, {"value": 3}]) == 5

def test_mean():
    assert mean([{"value": 2}, {"value": 4}]) == 3
TXT
printf '# Demo project\n\nRun `python main.py data.json`.\n\nTODO: add a license.\n' > /data/src/README.md
printf '[{"name": "a", "value": 2}, {"name": "b", "value": 5}, {"name": "c", "value": 11}]\n' > /data/src/data.json
printf 'secret=1\n' > /data/src/.env

head -c 2097152 /dev/zero | tr '\0' 'a' > /data/big/a.bin
head -c 614400 /dev/zero | tr '\0' 'b' > /data/big/b.bin
head -c 10240 /dev/zero | tr '\0' 'c' > /data/big/c.bin
printf 'small text file\n' > /data/big/d.txt

printf 'Quarterly report\nRevenue grew by 8 percent.\nCosts stayed flat.\n' > /data/docs/report.txt
printf '# Summary\n\n- revenue up\n- costs flat\n' > /data/docs/summary.md
printf 'obsolete draft\n' > /data/docs/sub/old.txt
printf 'body { margin: 0; }\n' > /data/site/css/main.css
printf '<html><body>Hello</body></html>\n' > /data/site/index.html
printf 'PNGDATA' > /data/site/img/logo.png
printf 'PNGDATA2' > /data/site/img/banner.png

( cd /data && tar -czf /data/backup.tar.gz docs )
( cd /data && zip -q -r /data/site.zip site )
chmod -R a+rX /data
find /data -exec touch -d '2026-01-15 10:00:00' {} +
