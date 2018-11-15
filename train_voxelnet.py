#!/usr/bin/env python3
import sys
sys.path.append('../aardvark')
sys.path.append('build/lib.linux-x86_64-' + sys.version[:3])
import random
import tensorflow as tf
import aardvark
import rpn3d            # read aardvark/rpn3d.py for BasicRPN3D
from zoo import net3d   # read aardvark/zoo/net3d.py
import cpp
from kitti import *

flags = tf.app.flags
FLAGS = flags.FLAGS

'''
flags.DEFINE_integer('max_points', 10000, '')

SAMPLE_DEPTH = 32
SAMPLE_SZ = 256
DOWNSIZE = 2

'''
T = 32
RANGES = [[-3, 1],
          [-40, 40],
          [0, 70.4]]
INPUT_SHAPE = [16, 400, 352]

class BasicVoxelNet (aardvark.Model, rpn3d.BasicRPN3D):
    # This is the base class of VoxelNet
    # this class sets up basic inference pipeline for Kitti-like
    # point cloud data:
    #   1. Input is a bunch of voxels, each voxel cointaining a variable
    #      number of points.
    #   2. A sub-network to extract features from each voxel
    #   3. A standard 3D CNN for box regression (via BasicRPN3D)

    # The actual implementation of rpn and vfe components are to be
    # implemented by a subclass.

    def __init__ (self):
        aardvark.Model.__init__(self)
        rpn3d.BasicRPN3D.__init__(self)
        pass

    def vfe (self, points, lengths):
        # points:       N * T * C
        # lengths:      N * 1
        assert False

    def build_graph (self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        Z, Y, X = INPUT_SHAPE
        V = Z * Y * X
        # 'points' grouped by voxels, each group should have been pre-capped at T
        #       with empty ones filled with 0s
        # then length of each group is given by 'lengths'
        self.points = tf.placeholder(tf.float32, name='points',
                                shape=(FLAGS.batch, V, T, FLAGS.channels))
        self.lengths = tf.placeholder(tf.float32, name='lengths',
                                shape=(FLAGS.batch, V, 1))
        flatten_points = tf.reshape(self.points, (FLAGS.batch * V, T, FLAGS.channels))
        flatten_lengths = tf.reshape(self.lengths, (FLAGS.batch * V, 1))
        # we send a batch of voxels into the vfe_net
        # the shape of the 3D grid shouldn't concern vfe_net
        net = self.vfe(flatten_points, flatten_lengths)
        net = tf.reshape(net, (FLAGS.batch, Z, Y, X, -1))

        self.build_rpn(net, self.is_training, shape=(INPUT_SHAPE))
        pass


class Stream:
    # Streaming samples, mimic picpac API
    def __init__ (self, path, priors, is_training):
        self.vxl = cpp.Voxelizer(np.array(RANGES, dtype=np.float32), np.array(INPUT_SHAPE, dtype=np.int32))
        self.priors = priors

        samples = []
        with open(path, 'r') as f:
            for l in f:
                samples.append(int(l.strip()))
                pass
            pass

        self.samples = samples
        self.sz = len(samples)
        self.is_training = is_training
        self.reset()
        pass

    def reset (self):
        samples = self.samples
        is_training = self.is_training

        def generator ():
            #conn = scipy.ndimage.generate_binary_structure(3, 2)
            while True:
                if is_training:
                    random.shuffle(samples)
                    pass
                for pk in samples:
                    sample = Sample(pk, LOAD_VELO | LOAD_CALIB | LOAD_LABEL2, is_training=True)
                    # note that Sample's is_training is True for both training and validation

                    meta = lambda: None
                    setattr(meta, 'ids', np.zeros((1,)))

                    assert sample.points.shape[1] == FLAGS.channels, 'channels should be %d.' % sample.points.shape[1]

                    points, lengths = self.vxl.voxelize_points([sample.points], T)

                    boxes = sample.get_boxes_array(["Car"])
                    anchors, anchors_weight, params, params_weight = self.vxl.voxelize_labels([boxes], self.priors, FLAGS.rpn_stride)

                    yield meta, points, lengths, anchors, anchors_weight, params, params_weight
                if not self.is_training:
                    break
        self.generator = generator()
        pass

    def size (self):
        return self.sz

    def next (self):
        return next(self.generator)


class VoxelNet (BasicVoxelNet):
    def __init__ (self):
        super().__init__()
        pass

    def vfe (self, net, lengths):
        # net: N * T * C
        # lengths: N * 1
        net = tf.reduce_sum(net, axis=1) / (lengths + 0.00001)
        return net

    def rpn_backbone (self, volume, is_training, stride):
        assert stride == 1
        net, st = net3d.unet(volume, is_training)
        return net

    def rpn_logits (self, net, is_training, channels):
        return tf.layers.conv3d(net, channels, 3, strides=1, activation=None, padding='SAME')

    def rpn_params (self, net, is_training, channels):
        return tf.layers.conv3d(net, channels, 3, strides=1, activation=None, padding='SAME')

    def create_stream (self, path, is_training):
        return Stream(path, self.priors, is_training)

    def feed_dict (self, record, is_training = True):
        _, pts, l, a, aw, p, pw = record
        return {self.points: pts,
                self.lengths: l,
                self.gt_anchors: a,
                self.gt_anchors_weight: aw,
                self.gt_params: p,
                self.gt_params_weight: pw,
                self.is_training: is_training}
    pass

def main (_):
    #./train.py --classes 2 --db data/multi_train_trim.list  --val_db data/multi_val_trim.list --epoch_steps 20 --ckpt_epochs 1 --val_epochs 1000
    FLAGS.channels = 4
    FLAGS.classes = None    # we shouldn't need this
    FLAGS.db = 'kitti_data/train.txt'
    FLAGS.val_db = 'kitti_data/val.txt'
    FLAGS.epoch_steps = 100
    FLAGS.ckpt_epochs = 1
    FLAGS.val_epochs = 1000
    FLAGS.model = "vxlnet"
    FLAGS.rpn_stride = 1
    model = VoxelNet()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

