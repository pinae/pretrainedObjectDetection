# -*- coding: utf-8 -*-
from scipy.io import loadmat
import json
import os


if __name__ == "__main__":
    x = loadmat(os.path.join(os.path.curdir, '..', '..', 'Datasets', 'ILSVRC2012', 'meta.mat'))
    labels_map = dict()
    for d in x['synsets']:
        labels_map[str(int(d[0][0]))] = {'id': d[0][1][0], 'label': d[0][2][0]}
    with open(os.path.join(os.path.curdir, 'ILSVRC2012_labels_map.json'), 'w') as json_file:
        json.dump(labels_map, json_file)
