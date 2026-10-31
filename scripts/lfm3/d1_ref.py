"""Transformers reference for the LiquidAI d1 CMF port: the checkpoint's own
System One code (trust_remote_code) over a battery of states and questions of
every type, in float32 (bf16 weights upcast) unless `bf16` is asked. Writes
the requests and the full answers (probabilities included) as JSON.

usage: python3 -I d1_ref.py MODEL_DIR {f32|bf16} OUT.json [IMAGE] [AUDIO.wav]
(d1-Omni: an AUDIO clip adds a spoken-request battery)
"""

import json
import sys

import torch
from transformers import AutoModel

TICKET = "I was charged twice and my parcel never arrived. Please refund one charge."
INVOICE = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)"
SUPPORT = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels",
                          "technical": "App or site faults"}},
    "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does the message raise?",
               "criteria": {"billing": "Payments and charges", "delivery": "Shipping and parcels",
                            "login": "Signing in to the account"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]},
    "angry": {"type": "noul", "instructions": "Is the customer angry?",
              "criteria": {"true": "insults, threats or shouting", "false": "calm or merely annoyed"}},
}
EXTRACT = {
    "payee": {"type": "text", "instructions": "Who issued the invoice?"},
    "total": {"type": "number", "instructions": "How much is due?"},
    "due": {"type": "date", "instructions": "By when must it be paid?"},
    "vat": {"type": "number", "instructions": "How much VAT is charged?"},
    "lines": {"type": "number", "instructions": "How many lines does the invoice have?", "range": [0, 10], "step": 1},
}
JSON_STATE = {"customer": {"name": "Maria Lopez", "tier": "gold"},
              "messages": ["Hi, my order 4471 arrived broken.", "Can I get a replacement by Friday?"]}
JSON_Q = {
    "name": {"type": "text", "instructions": "What is the customer's name?"},
    "order": {"type": "number", "instructions": "What is the order number?"},
    "replace": {"type": "noul", "instructions": "Does the customer want a replacement?"},
    "sentiment": {"type": "score", "instructions": "How satisfied is the customer?",
                  "criteria": ["very unhappy", "unhappy", "neutral", "happy", "very happy"]},
}
AUDIO_Q = {
    "booking": {"type": "noul", "instructions": "Is the speaker making a reservation?"},
    "people": {"type": "number", "instructions": "For how many people?", "range": [0, 10], "step": 1},
    "kind": {"type": "choice", "instructions": "What is the request about?",
             "criteria": {"restaurant": "A table or a meal", "travel": "Flights or hotels",
                          "support": "A product problem"}},
}
IMAGE_Q = {
    "cats": {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
    "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
    "every_cat": {"type": "boxes", "instructions": "Every cat.", "max": 4},
    "indoor": {"type": "noul", "instructions": "Is this picture taken indoors?"},
    "color": {"type": "choice", "instructions": "What color is the sofa?",
              "criteria": {"pink": None, "blue": None, "green": None, "black": None}},
}


def main():
    model_dir, dtype, out = sys.argv[1], sys.argv[2], sys.argv[3]
    image = sys.argv[4] if len(sys.argv) > 4 else None
    audio = sys.argv[5] if len(sys.argv) > 5 else None
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                      dtype={"bf16": torch.bfloat16, "f32": torch.float32}[dtype]).to("cuda:0")
    requests = [(TICKET, SUPPORT, None), (INVOICE, EXTRACT, None), (JSON_STATE, JSON_Q, None)]
    if image:
        from PIL import Image
        requests.append((None, IMAGE_Q, [Image.open(image).convert("RGB")]))
    if audio:
        requests.append((None, AUDIO_Q, [{"type": "audio", "path": audio}]))
    records = []
    for state, qs, media in requests:
        res = model.system_one(state, qs, media=media)
        print(json.dumps(res["answers"], ensure_ascii=False)[:600], flush=True)
        is_audio = bool(media) and isinstance(media[0], dict)
        records.append({"state": state, "questions": qs, "image": image if media and not is_audio else None,
                        "audio": audio if is_audio else None, "result": res})
    with open(out, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
