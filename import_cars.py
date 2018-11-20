#!/usr/bin/env python3
import sys
import h5py
from tqdm import tqdm
from kitti import *
from train_voxelnet import *

def import_sample (pk, root):
    sample = Sample(pk, LOAD_VELO | LOAD_LABEL2, True)
    points = sample.get_voxelnet_points()
    assert points.shape[1] == 4, 'channels should be %d.' % sample.points.shape[1]
    # we might want to filter the points by range
    boxes = sample.get_voxelnet_boxes(["Car"])

    with h5py.File(os.path.join(root, '%06d.h5' % pk), "w") as f:
        f.create_dataset("points", data=points)
        f.create_dataset("boxes", data=boxes)
        pass
    pass

def import_set (list_path):
    print("Importing", list_path)
    root = os.path.join(DATA_ROOT, 'training', 'cars.h5')
    try:
        os.makedirs(root)
    except:
        pass

    tasks = []
    with open(list_path, 'r') as f:
        for l in f:
            pk = int(l.strip())
            tasks.append(pk)
            pass
        pass
    for pk in tqdm(tasks):
        import_sample(pk, root)
        pass
    pass

import_set('kitti_data/trainval.txt')

