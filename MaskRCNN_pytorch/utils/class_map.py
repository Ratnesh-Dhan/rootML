import json

with open("/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/meta.json") as f:
    meta = json.load(f)

class_map = {
    cls["title"]: idx + 1
    for idx, cls in enumerate(meta["classes"])
}

print(class_map)