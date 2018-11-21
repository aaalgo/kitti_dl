#!/usr/bin/env python3
import sys
sys.path.append('../aardvark')  # git clone https://github.com/aaalgo/aardvark
sys.path.append('build/lib.linux-x86_64-' + sys.version[:3])
import random
import tensorflow as tf
import aardvark
import rpn                      # read aardvark/rpn.py for RPN
import cpp
from kitti import *

flags = tf.app.flags
flags.DEFINE_string('test_db', None, 'test db')
                                # db and val_db defined in aardvark.py
FLAGS = flags.FLAGS

T = 35                          # maximal T points per voxel
                                # TODO: should be 35, fix in next train

# The setting is for Cars only.
RANGES = np.array([[0, 70.4],   # X: back -> front
                   [-40, 40],   # Y: left -- right
                   [-1, 3]], dtype=np.float32)
                                # Z: ground -- sky
INPUT_SHAPE = np.array([352, 400, 8], dtype=np.int32)
                                # shape of Z should be 10 according to paper


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

    def size (self):
        return self.sz

    def next (self):
        return self.impl.next()

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
                    yield ['kitti_data/training/cars.h5/%06d.h5' % pk]
                if not self.is_training:
                    break
        self.impl = cpp.Streamer(generator(), RANGES, INPUT_SHAPE, np.array(self.priors, dtype=np.float32), FLAGS.rpn_stride, T, FLAGS.lower_th, FLAGS.upper_th)
        pass

def conv2d (net, ch, kernel, strides, is_training):
    net = tf.layers.conv2d(net, ch, kernel, strides, padding='same')
    net = tf.layers.batch_normalization(net, training=is_training)
    return tf.nn.relu(net)

def deconv2d (net, ch, kernel, strides, is_training):
    net = tf.layers.conv2d_transpose(net, ch, kernel, strides, padding='same')
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    return net

def conv3d (net, ch, kernel, strides, is_training, padding='same'): 
    net = tf.layers.conv3d(net, ch, kernel, strides, padding=padding)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    return net

class VoxelNet (rpn.RPN):

    def __init__ (self):
        super().__init__()
        self.vxl = cpp.Voxelizer(RANGES, INPUT_SHAPE)
        pass

    def vfe (self, net, mask):
        # Stacked Voxel Feature Encoding
        net = tf.expand_dims(net, axis=0)   # expand so we can use 2D 1x1 convolution
        # net:  1 * V * T * C               # to achieve "FCN" as described in paper.
        # mask: 1 * V * T * 1               # "FCN" in paper is a wrong term.
        # Actual # points in voxel is <= T
        # If there are N points the first N entry in mask is 1 and others 0.
        for ch in [32, 128]:
            net = conv2d(net, ch//2, 1, 1, self.is_training)    # "FCN"
            # net: B * V * T * C'
            net = net * mask
            pool = tf.reduce_max(net, axis=2, keepdims=True)
            # net: B * V * 1 * C'
            pool = tf.tile(pool, [1, 1, T, 1])
            net = tf.concat([net, pool], axis=3)
            pass
        # voxel feature extraction
        net = conv2d(net, 128, 1, 1, self.is_training)
        net = net * mask
        net = tf.reduce_max(net, axis=2)
        net = tf.squeeze(net, axis=0)
        return net

    def middle (self, net):
        # Section 3.1.                                          #                   K  strides  padding
        net = conv3d(net, 64, 3, (1,1,2), self.is_training)     # paper: Conv3D(64, 3, (2,1,1), (1,1,1))
        net = conv3d(net, 64, 3, (1,1,1), self.is_training)     # paper: Conv3D(64, 3, (1,1,1), (0,1,1))
        net = conv3d(net, 64, 3, (1,1,2), self.is_training)     # paper: Conv3D(64, 3, (2,1,1), (1,1,1))
        return net

    def rpn_backbone (self, net):
        # block1
        net = conv2d(net, 128, 3, 2, self.is_training)
        for _ in range(3):
            net = conv2d(net, 128, 3, 1, self.is_training)
        upscale1 = deconv2d(net, 256, 3, 1, self.is_training)

        # block2
        net = conv2d(net, 128, 3, 2, self.is_training)
        for _ in range(5):
            net = conv2d(net, 128, 3, 1, self.is_training)
        upscale2 = deconv2d(net, 256, 2, 2, self.is_training)

        net = conv2d(net, 256, 3, 2, self.is_training)
        for _ in range(5):
            net = conv2d(net, 256, 3, 1, self.is_training)
        upscale3 = deconv2d(net, 256, 4, 4, self.is_training)
        net = tf.concat([upscale1, upscale2, upscale3], axis=3)
        self.backbone = net
        pass

    def rpn_logits (self, channels, strides):
        assert strides == 2
        return tf.layers.conv2d(self.backbone, channels, 1, 1)      # do we need BN here?

    def rpn_params (self, channels, strides):
        return tf.layers.conv2d(self.backbone, channels, 1, 1)      # do we need BN here?

    def rpn_params_size (self):
        return 8        # 

    def rpn_params_loss (self, params, gt_params, priors):
        # params        ? * priors * 7
        # gt_params     ? * priors * 7
        # priors        1 * priors * 2

        #gt_params = gt_params / priors
        l1 = tf.losses.huber_loss(params, gt_params, reduction=tf.losses.Reduction.NONE, loss_collection=None)
        return tf.reduce_sum(l1, axis=2)

    def rpn_generate_shapes (self, shape, anchor_params, priors, n_priors):
        return None

    def build_graph (self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        X, Y, Z = INPUT_SHAPE
        V = Z * Y * X
        self.points = tf.placeholder(tf.float32, name='points', shape=(None, T, FLAGS.channels))
        self.mask = tf.placeholder(tf.float32, name='mask', shape=(None, T, 1))
        self.index = tf.placeholder(tf.int32, name='index', shape=(None,))

        # we send a batch of voxels into the vfe_net
        # the shape of the 3D grid shouldn't concern vfe_net
        net = self.vfe(self.points, self.mask)
        # conver sparse voxels into dense 3D volume
        net, = tf.py_func(self.vxl.make_dense, [net, self.index], [tf.float32])
        net = tf.reshape(net, (FLAGS.batch, X, Y, Z, 128))

        self.voxel_features = net
        net = self.middle(net)
        net = tf.reshape(net, (FLAGS.batch, X, Y, -1))

        # see aardvark/rpn.py for details
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

def setup_params ():
    FLAGS.channels = 7
    FLAGS.classes = None    # we shouldn't need this
    FLAGS.db = 'kitti_data/train.txt'
    FLAGS.val_db = 'kitti_data/val.txt'
    FLAGS.test_db = 'kitti_data/test.txt'
    FLAGS.epoch_steps = 100
    FLAGS.ckpt_epochs = 1
    FLAGS.val_epochs = 10000
    FLAGS.rpn_stride = 2
    FLAGS.lower_th = 0.15 #0.45
    FLAGS.upper_th = 0.60
    FLAGS.rpn_positive_extra = 0.5
    FLAGS.decay_steps = 1000
    FLAGS.nms_th = 0.2
    pass

def main (_):
    setup_params()
    FLAGS.model = "vxlnet"
    model = VoxelNet()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

