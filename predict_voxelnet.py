#!/usr/bin/env python3
from train_voxelnet import *
import subprocess as sp
from gallery import Gallery

flags = tf.app.flags
flags.DEFINE_integer('max', 20, '')
flags.DEFINE_string('gallery', None, '')
flags.DEFINE_bool('test_labels', None, '')
flags.DEFINE_string('results', None, '')
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
        pass

    if FLAGS.results:
        sp.check_call('mkdir -p %s/data' % FLAGS.results, shell=True)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if not FLAGS.test_labels:
            saver.restore(sess, FLAGS.model)

        gal = Gallery('output', cols=columns)
        C = 0
        with open(db, 'r') as f:
            for l in f:
                pk = int(l.strip())
                sample = Sample(pk, LOAD_IMAGE2 | LOAD_VELO | LOAD_LABEL2, is_training=is_val)
                points = sample.get_voxelnet_points()
                points, mask, index = model.vxl.voxelize_points([points], T)
                feed_dict = {model.is_training: False,
                             model.points: points,
                             model.mask: mask,
                             model.index: index}

                if is_val:
                    # for validation set, produce the ground-truth boxes
                    boxes_gt = sample.get_voxelnet_boxes(["Car"])
                    #for row in boxes_gt:
                    #    print(row)
                    if FLAGS.test_labels:   # 2 lines below are for testing the C++ code
                        probs, _, params, _ = model.vxl.voxelize_labels([boxes_gt], np.array(model.priors, dtype=np.float32), FLAGS.rpn_stride, FLAGS.lower_th, FLAGS.upper_th)
                    sample.load_voxelnet_boxes(boxes_gt, 'Car')
                    # visualize groundtruth labels
                    image3d = np.copy(sample.image2)
                    for box in sample.label2:
                        draw_box3d(image3d, box, sample.calib)
                        pass
                    cv2.imwrite(gal.next(), image3d)

                if not FLAGS.test_labels:
                    probs, params = sess.run([model.probs, model.params], feed_dict=feed_dict)

                boxes = model.vxl.generate_boxes(probs, params, np.array(model.priors, dtype=np.float32), FLAGS.anchor_th)
                boxes = cpp.nms(boxes, FLAGS.nms_th)
                boxes = boxes[0]
                #print("++++")
                #for row in boxes:
                #    print(row)
                #print('====')
                print(np.max(probs), len(boxes))
                sample.load_voxelnet_boxes(boxes, 'Car')

                if FLAGS.results:
                    save_label2('%s/data/%06d.txt' % (FLAGS.results, pk), sample.label2)

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


