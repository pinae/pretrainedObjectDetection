# -*- coding: utf-8 -*-
from tensorflow.data import Dataset
from tensorflow import uint32, uint8, TensorShape
from model import prepare_input
import os
import cv2


def data_generator():
    i = 0
    dataset_dir = os.path.join(os.path.curdir, "..", "..", "Datasets", "ILSVRC2012")
    with open(os.path.join(dataset_dir, "ILSVRC2012_validation_ground_truth.txt"), 'r') as f:
        labels = f.readlines()
    while i < len(labels):
        img_filename = "ILSVRC2012_val_000%05d.JPEG" % (i + 1)
        img_path = os.path.join(dataset_dir, "images", img_filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_AREA)
        label = int(labels[i])
        yield (img, label)
        i += 1


def get_dataset():
    return Dataset.from_generator(
           data_generator,
           output_types=(uint8, uint32),
           output_shapes=(TensorShape([299, 299, 3]), TensorShape([])))


def get_representative_data():
    representative_data = []
    for i, (img, label) in enumerate(data_generator()):
        if i > 10:
            break
        representative_data.append([prepare_input(img)])
    return representative_data


if __name__ == "__main__":
    d = get_dataset()
    for x in d.take(3):
        print(x)
