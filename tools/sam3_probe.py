#!/usr/bin/env python3
"""Print the attribute structure of transformers Sam3Model (env: HF_TOKEN)."""
import os
from transformers import Sam3Model

m = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
print("CHILDREN:", list(dict(m.named_children()).keys()))
print("ATTRS:", [a for a in dir(m) if any(s in a.lower()
      for s in ("det", "back", "trans", "image", "text", "seg"))])
