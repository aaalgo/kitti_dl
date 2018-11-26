# kitti_dl
KITTI 3D solution by voxelnet


1. Preparation

The projects depends on boost 1.66+ and a recent version of Tensorflow.


```
git clone https://github.com/aaalgo/kitti_dl
cd kitt_dl
./setup.sh
```

Download the Kitti velodyne data and setup symbolic links such that
the following files exist.

```
kitti_data/training/velodyne
├── 000000.bin
├── 000001.bin

```

2. Preprocess Data
```
./import_cars.py
```

3. Train
```
./train_voxelnet.py 
```
Models will be saved in the directory vxlnet.

