#!/usr/bin/env python3
from train_voxelnet import *
from gallery import Gallery

flags = tf.app.flags
flags.DEFINE_integer('max', 20, '')
flags.DEFINE_string('gallery', None, '')
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
        columns = 2 # two columns to visualize groundtruth
    else:
        db = FLAGS.test_db
        columns = 1

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, FLAGS.model)

        gal = Gallery('output', cols=columns)
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

                if is_val:
                    # for validation set, produce the ground-truth boxes
                    boxes_gt = sample.get_boxes_array(["Car"])
                    if False:   # 2 lines below are for testing the C++ code
                        probs_gt, _, params_gt, _ = model.vxl.voxelize_labels([boxes_gt], np.array(model.priors, dtype=np.float32), FLAGS.rpn_stride)
                        boxes_bt = model.vxl.generate_boxes(probs_gt, params_gt, FLAGS.anchor_th)
                    sample.load_boxes_array(boxes_gt, 'Car')
                    # visualize groundtruth labels
                    image3d = np.copy(sample.image2)
                    for box in sample.label2:
                        draw_box3d(image3d, box, sample.calib)
                        pass
                    cv2.imwrite(gal.next(), image3d)

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


