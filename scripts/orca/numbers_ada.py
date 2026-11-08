# Measured on the RTX 2000 Ada pod (16 GB, 70 W), cortiq 0.8.63 + infosave2007/cmf#99, Vulkan:
# `bench --core --tokens 60 --ignore-eos --ctx N` (logs/final.log). Memory: nvidia-smi's peak in MiB / 1024.
# The standard file does not fit the card from 16 000 tokens on (out of device memory): None.
CTX = [1000, 4000, 16000, 32000, 64000]
LABELS = ["1 000", "4 000", "16 000", "32 000", "64 000"]
STD = dict(dec=[27.4, 25.8, None, None, None], first=[9.14, 36.99, None, None, None], vram=[14.13, 14.53, None, None, None])
TUR = dict(dec=[29.9, 26.7, 26.7, 26.3, 26.0], first=[8.89, 36.69, 159.13, 323.78, 652.47], vram=[14.13, 14.53, 14.75, 14.75, 14.77])
SCALE = dict(dec=(35, 5, (30, -16)), first=(700, 100, (-16, 34)), vram=(18, 2, (-16, 34)))
TXT = {
    "en": {"res": "14.8 GB of device memory at 16 000, 32 000 and 64 000 tokens; the standard file does not fit the 16 GB card from 16 000",
           "nofit": "standard: out of device memory from 16 000"},
    "ru": {"res": "14,8 ГБ памяти карты на 16 000, 32 000 и 64 000 токенов; обычный файл не помещается на карту 16 ГБ с 16 000",
           "nofit": "обычный: не хватает памяти карты с 16 000"},
    "zh": {"res": "16 000、32 000 和 64 000 token 时显存均为 14.8 GB；普通文件从 16 000 起放不进 16 GB 显卡",
           "nofit": "普通文件：从 16 000 起显存不足"},
}
