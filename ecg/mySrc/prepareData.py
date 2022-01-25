from fileinput import filename
import json
import numpy as np
import os

# index = 3

CLASS_MAPPER = json.load(open("../data/class-mapper.json"))
# print(CLASS_MAPPER)
ann_path = "../data/val.json"
data = json.load(open(ann_path))
# print(data[index])

for index in range(len(data)):
    img = np.load(data[index]["path"]).astype("float32")
    img = img.reshape(1, 1, img.shape[0])
    dir = CLASS_MAPPER[data[index]["label"]]

    os.makedirs(f"./val_data/{dir}", exist_ok=True)
    name = data[index]["name"]
    filename = data[index]["filename"]
    savename = f"./val_data/{dir}/{name}_{filename}.npy"
    np.save(savename, img)
