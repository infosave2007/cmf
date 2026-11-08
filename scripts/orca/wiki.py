#!/usr/bin/env python3
# wikitext-2-raw-v1 test split -> wiki_test.txt (the rows joined as in wiki.test.raw). wiki.py OUT
import sys, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
t = pq.read_table(p).column("text").to_pylist()
open(sys.argv[1], "w", encoding="utf-8").write("".join(t))
print(len(t), "rows", sum(len(x) for x in t), "chars")
