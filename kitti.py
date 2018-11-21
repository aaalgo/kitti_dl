#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import cv2

DATA_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kitti_data')

LOAD_IMAGE2 = 0x01
LOAD_IMAGE3 = 0x02
LOAD_VELO = 0x04
LOAD_LABEL2 = 0x8

LOAD_ALL = LOAD_IMAGE2 | LOAD_IMAGE3 | LOAD_VELO | LOAD_LABEL2

TYPES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

def empty_object ():
    return lambda: None

def load_label2 (path):
    # load label file, see kitti devkit readme.txt for doc
    # example:
    # Pedestrian 0.00 0 0.18 409.32 149.33 490.32 308.16 1.82 0.63 1.14 -1.80 1.46 8.47 -0.02
    objs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            obj = empty_object()
            obj.type = line[0]
            obj.truncated = float(line[1])
            obj.occluded = int(line[2])
            obj.alpha = float(line[3])
                # Observation angle of object, ranging [-pi..pi]
            obj.bbox = [float(x) for x in line[4:8]]
                # 2D bounding box, left, top, right bottom
            obj.dim = [float(x) for x in line[8:11]]    # h, w, l
                # height, width, length in meters
            obj.loc = [float(x) for x in line[11:14]]   # x, y, z
                # x, y, z in camera coordinate in meters
            obj.rot = float(line[14])                   # ry
                # rotation ry around Y-axis in camera coordinates [-pi..pi]
            obj.score = 1.0 #float(line[15])                   # ry
            objs.append(obj)
            pass
        pass
    return objs

def box3d_corners (obj):
    # get the 3D box corners of an object in camera space
    idx8 = [[0,0,0], [0,0,1], [0,1,1], [0,1,0],
            [1,0,0], [1,0,1], [1,1,1], [1,1,0]]
    c = np.cos(obj.rot)
    s = np.sin(obj.rot)
    rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

    points = []
    for d1, d2, d3 in idx8:
        h, w, l = obj.dim
        x, y, z = 0, 0, 0
        x -= l/2
        z -= w/2
        points.append([x + d1 * l, y - d2 * h, z + d3 * w])
    points = np.array(points, dtype=np.float32) @ rot
    points += np.array([obj.loc], dtype=np.float32)
    return points

def load_calib_line (f, name, shape):
    line = f.readline().strip().split(' ')
    assert line[0] == name + ':' 
    arr = np.array([float(x) for x in line[1:]], dtype=np.float32)
    return np.reshape(arr, shape)

def load_calib (path):
    with open(path, 'r') as f:
        obj = empty_object()
        obj.P0 = load_calib_line(f, 'P0', (3,4))
        obj.P1 = load_calib_line(f, 'P1', (3,4))
        obj.P2 = load_calib_line(f, 'P2', (3,4))
        obj.P3 = load_calib_line(f, 'P3', (3,4))
        obj.R0_rect = np.eye(4, dtype=np.float32)
        obj.R0_rect[:3, :3] = load_calib_line(f, 'R0_rect', (3,3))
        obj.Tr_velo_to_cam = np.eye(4, dtype=np.float32)
        obj.Tr_velo_to_cam[:3, :4] = load_calib_line(f, 'Tr_velo_to_cam', (3,4))
        obj.Tr_imu_to_velo = np.eye(4, dtype=np.float32)
        obj.Tr_imu_to_velo[:3, :4] = load_calib_line(f, 'Tr_imu_to_velo', (3,4))
        return obj

def load_points (path, columns=4, mapping=None):
    pc = np.fromfile(path, dtype=np.float32)
    pc = np.reshape(pc, (-1, columns))
    if not mapping is None:
        pc = pc[:, mapping]
        pass
    return pc


class Sample:
    def __init__ (self, pk, load_flags = LOAD_ALL, is_training = True):
        sub = 'testing'
        if is_training:
            sub = 'training'
            pass
        root = os.path.join(DATA_ROOT, sub)

        if load_flags | LOAD_IMAGE2:
            self.image2 = cv2.imread(os.path.join(root, 'image_2/%06d.png' % pk), cv2.IMREAD_COLOR)
            pass
        if load_flags | LOAD_IMAGE3:
            self.image3 = cv2.imread(os.path.join(root, 'image_3/%06d.png' % pk), cv2.IMREAD_COLOR)
            pass
        if load_flags | LOAD_VELO:
            self.calib = load_calib(os.path.join(root, 'calib/%06d.txt' % pk))
            self.points = load_points(os.path.join(root, 'velodyne/%06d.bin' % pk))
            #points = points @ self.calib.Tr_velo_to_cam.T
            pass
        if load_flags | LOAD_LABEL2:
            self.label2 = load_label2(os.path.join(root, 'label_2/%06d.txt' % pk))
            pass
        pass

    # voxel net use different meaning of X Y Z
    # and axis need to be swapped
    def get_voxelnet_points (self):
        points = self.points
        C3 = points[:, 3]
        points[:, 3] = 1.0
        points = points @ self.calib.Tr_velo_to_cam.T
        points[:, 3] = C3;
        points = np.copy(points[:, [2, 0, 1, 3]].astype(np.float32), order='C')
        return points

    def get_voxelnet_boxes (self, types):

        boxes = []
        for obj in self.label2:
            if obj.type in types:
                x, y, z = obj.loc
                h, w, l = obj.dim
                boxes.append([z, x, y, h, w, l, obj.rot, obj.score])
                pass
            pass
        if len(boxes) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)

    def load_voxelnet_boxes (self, array, type1):
        boxes = []
        for row in array:
            box = empty_object()
            box.type = type1
            print(row)
            z, x, y, h, w, l, box.rot, box.score = row
            box.loc = (x, y, z)
            box.dim = (h, w, l)
            boxes.append(box)
            pass
        self.label2 = boxes
    pass

def draw_box3d (image, obj, calib):
    points = box3d_corners(obj)
    x = np.ones((points.shape[0], 4), dtype=np.float32)
    x[:, :3] = points
    points = x
    points = points @ (calib.P2).T
    points[:, 0] /= points[:, 2]
    points[:, 1] /= points[:, 2]
    points = np.round(points[:, :2]).astype(np.int32)

    pts = []
    for x, y in points:
        pts.append((x,y))

    lines = [(0, 1), (1,2), (2,3), (3,0),
             (4, 5), (5,6), (6,7), (7,4),
             (0, 4), (1,5), (2,6), (3,7)]
    for i1, i2 in lines[:4]:
        cv2.line(image, pts[i1], pts[i2], (0, 0, 255))
    for i1, i2 in lines[4:]:
        cv2.line(image, pts[i1], pts[i2], (255, 0, 0))
    for i1, i2 in lines[8:]:
        cv2.line(image, pts[i1], pts[i2], (0, 255, 0))

    #left, top = [int(round(x)) for x in obj.bbox[:2]]
    #cv2.putText(image, obj.type[:4], (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    pass

if __name__ == '__main__':
    sample = Sample(200)
    image2d = np.copy(sample.image2)
    image3d = np.copy(sample.image2)
    for obj in sample.label2:
        # draw 2D bounding boxes
        left, top, right, bottom = [int(round(x)) for x in obj.bbox]
        cv2.rectangle(image2d, (left, top), (right, bottom), (0, 255,0))
        cv2.putText(image2d, obj.type[:4], (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

        # draw 3D bounding box
        draw_box3d(image3d, obj, sample.calib)
        pass
    cv2.imwrite('test.png', np.vstack([image2d, image3d]))

