#!/usr/bin/env python3
from train_voxelnet import *
from gallery import Gallery

flags = tf.app.flags
flags.DEFINE_integer('max', 10, '')
FLAGS = flags.FLAGS

def main (_):
    setup_params()
    model = VoxelNet()
    model.build_graph()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    is_val = True
    if is_val:
        db = FLAGS.val_db
    else:
        db = FLAGS.test_db

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, FLAGS.model)

        gal = Gallery('output')
        C = 0
        with open(db, 'r') as f:
            for l in f:
                pk = int(l.strip())
                sample = Sample(pk, LOAD_IMAGE2 | LOAD_VELO | LOAD_LABEL2, is_training=is_val)
                points = sample.get_points_swapped()
                points, mask, index = model.vxl.voxelize_points([points], T)
                feed_dict = {model.is_training: False,
                             model.points: points,
                             model.mask: mask,
                             model.index: index}

                if False:
                    # test if probs and params generation in training is correct
                    boxes = sample.get_boxes_array(["Car"])
                    probs, _, params, _ = model.vxl.voxelize_labels([boxes], np.array(model.priors, dtype=np.float32), FLAGS.rpn_stride)
                else:
                    probs, params = sess.run([model.probs, model.params], feed_dict=feed_dict)
                boxes = model.vxl.generate_boxes(probs, params, FLAGS.anchor_th)
                boxes = boxes[0]
                print(np.max(probs), len(boxes))
                sample.load_boxes_array(boxes, 'Car')

                image3d = np.copy(sample.image2)
                for box in sample.label2:
                    draw_box3d(image3d, box, sample.calib)
                    pass
                cv2.imwrite(gal.next(), image3d)
                C += 1
                if C >= FLAGS.max:
                    break
                pass
            pass
        gal.flush()
        pass
    pass


if __name__ == '__main__':
    tf.app.run()


