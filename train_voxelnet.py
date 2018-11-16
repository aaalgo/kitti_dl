#!/usr/bin/env python3
import sys
sys.path.append('../aardvark')
sys.path.append('build/lib.linux-x86_64-' + sys.version[:3])
import random
import tensorflow as tf
import aardvark
import rpn            # read aardvark/rpn.py for RPN
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
RANGES = [[0, 70.4],    # X
          [-40, 40],    # Y
          [-3, 1]]      # Z
INPUT_SHAPE = [352, 400, 8]


class Stream:
    # Streaming samples, mimic picpac API
    def __init__ (self, path, vxl, priors, is_training):
        self.vxl = vxl
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

                    assert sample.points.shape[1] == 4, 'channels should be %d.' % sample.points.shape[1]

                    points, mask, index = self.vxl.voxelize_points([sample.points], T)

                    boxes = sample.get_boxes_array(["Car"])
                    anchors, anchors_weight, params, params_weight = self.vxl.voxelize_labels([boxes], np.array(self.priors, dtype=np.float32), FLAGS.rpn_stride)

                    yield meta, points, mask, index, anchors, anchors_weight, params, params_weight
                if not self.is_training:
                    break
        self.generator = generator()
        pass

    def size (self):
        return self.sz

    def next (self):
        return next(self.generator)


def conv2d (net, ch, kernel, strides, is_training):
    net = tf.layers.conv2d(net, ch, kernel, strides, padding='same')
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    return net

def deconv2d (net, ch, kernel, strides, is_training):
    net = tf.layers.conv2d_transpose(net, ch, kernel, strides, padding='same')
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    return net

def conv3d (net, ch, kernel, strides, is_training): 
    net = tf.layers.conv3d(net, ch, kernel, strides, padding='same')
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    return net

class VoxelNet (rpn.RPN):
    # This is the base class of VoxelNet
    # this class sets up basic inference pipeline for Kitti-like
    # point cloud data:
    #   1. Input is a bunch of voxels, each voxel cointaining a variable
    #      number of points.
    #   2. A sub-network to extract features from each voxel #   3. A standard 3D CNN for box regression (via BasicRPN3D)

    # The actual implementation of rpn and vfe components are to be
    # implemented by a subclass.

    def __init__ (self):
        super().__init__()
        self.vxl = cpp.Voxelizer(np.array(RANGES, dtype=np.float32), np.array(INPUT_SHAPE, dtype=np.int32))
        pass

    def vfe (self, net, mask):
        net = tf.expand_dims(net, axis=0)
        # net: 1 * V * T * C
        # lengths: 1 * V * T * 1
        for ch in [32, 128]:
            net = conv2d(net, ch//2, 1, 1, self.is_training)
            # net: B * V * T * C'
            net = net * mask
            pool = tf.reduce_max(net, axis=2, keepdims=True)
            # net: B * V * 1 * C'
            pool = tf.tile(pool, [1, 1, T, 1])
            net = tf.concat([net, pool], axis=3)

        net = conv2d(net, 128, 1, 1, self.is_training)
        net = net * mask
        net = tf.reduce_max(net, axis=2)
        net = tf.squeeze(net, axis=0)
        return net

    def middle (self, net):
        # input is 352 x 400 x 10 x ?
        #
        for ch, str3 in [(64, 2), (64, 1), (64, 2)]:
            net = conv3d(net, ch, 3, (1,1,str3), self.is_training)
        return net

    def rpn_backbone (self, net):
        is_training = self.is_training
        net = conv2d(net, 128, 3, 2, is_training)
        for _ in range(3):
            net = conv2d(net, 128, 3, 1, is_training)
        block1 = net
        net = conv2d(net, 128, 3, 2, is_training)
        for _ in range(5):
            net = conv2d(net, 128, 3, 1, is_training)
        block2 = net
        net = conv2d(net, 256, 3, 2, is_training)
        for _ in range(5):
            net = conv2d(net, 256, 3, 1, is_training)
        net = deconv2d(net, 256, 4, 4, is_training)
        net2 = deconv2d(block2, 256, 2, 2, is_training)
        net1 = deconv2d(block1, 256, 3, 1, is_training)
        net = tf.concat([net, net1, net2], axis=3)
        self.backbone = net
        pass

    def rpn_logits (self, channels, strides):
        assert strides == 2
        return tf.layers.conv2d(self.backbone, channels, 1, strides=1, activation=None, padding='SAME')

    def rpn_params (self, channels, strides):
        return tf.layers.conv2d(self.backbone, channels, 1, strides=1, activation=None, padding='SAME')

    def rpn_non_max_supression (self, boxes, index, prob):
        nms_max = tf.constant(FLAGS.nms_max, dtype=tf.int32, name="nms_max")
        nms_th = tf.constant(FLAGS.nms_th, dtype=tf.float32, name="nms_th")
        return tf.image.non_max_suppression(rpn.shift_boxes(boxes, index), prob, nms_max, iou_threshold=nms_th)

    def build_graph (self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        X, Y, Z = INPUT_SHAPE
        V = Z * Y * X
        # 'points' grouped by voxels, each group should have been pre-capped at T
        #       with empty ones filled with 0s
        # then length of each group is given by 'lengths'
        self.points = tf.placeholder(tf.float32, name='points',
                                shape=(None, T, FLAGS.channels))
        self.mask = tf.placeholder(tf.float32, name='mask',
                                shape=(None, T, 1))
        self.index = tf.placeholder(tf.int32, name='index',
                                shape=(None,))
        # we send a batch of voxels into the vfe_net
        # the shape of the 3D grid shouldn't concern vfe_net
        net = self.vfe(self.points, self.mask)
        net, = tf.py_func(self.vxl.make_dense, [net, self.index], [tf.float32])
        net = tf.reshape(net, (FLAGS.batch, X, Y, Z, 128))

        net = self.middle(net)
        net = tf.reshape(net, (FLAGS.batch, X, Y, -1))

        self.build_rpn(net)
        pass

    def create_stream (self, path, is_training):
        return Stream(path, self.vxl, self.priors, is_training)

    def feed_dict (self, record, is_training = True):
        _, pts, mask, index, a, aw, p, pw = record
        return {self.points: pts,
                self.mask: mask,
                self.index: index,
                self.gt_anchors: a,
                self.gt_anchors_weight: aw,
                self.gt_params: p,
                self.gt_params_weight: pw,
                self.is_training: is_training}
    pass


def main (_):
    #./train.py --classes 2 --db data/multi_train_trim.list  --val_db data/multi_val_trim.list --epoch_steps 20 --ckpt_epochs 1 --val_epochs 1000
    FLAGS.channels = 7
    FLAGS.classes = None    # we shouldn't need this
    FLAGS.db = 'kitti_data/train.txt'
    FLAGS.val_db = 'kitti_data/val.txt'
    FLAGS.epoch_steps = 100
    FLAGS.ckpt_epochs = 1
    FLAGS.val_epochs = 1000
    FLAGS.model = "vxlnet"
    FLAGS.rpn_stride = 2
    model = VoxelNet()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

