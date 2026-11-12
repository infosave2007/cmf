# wikitext-2 raw test split as one text file (the passkey and perplexity runs read it).
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
t = pq.read_table(p).column("text").to_pylist()
open("/root/cuda/data/wiki.test.raw", "w", encoding="utf-8").write("".join(t))
