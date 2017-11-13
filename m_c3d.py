from m_video import clipread, randcrop
from sklearn.utils import shuffle
from time import ctime
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

HEIGHT = 128
WIDTH = 171
FRAMES = 16
CROP_SIZE = 112
CHANNELS = 3
BATCH_SIZE = 32


def c3d(inputs, num_class, training):
    """
    C3D network for video classification
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param num_class: A scalar for number of class
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :return: Output tensor
    """

    net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  # manually padding here :)

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID')
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  # another manually padding here ;)
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID')
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu)
    net = tf.identity(net, name='fc1')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu)
    net = tf.identity(net, name='fc2')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=num_class, activation=None)
    net = tf.identity(net, name='logits')
    return net


def c3d_ucf101(inputs, training, weights=None):
    """
    C3D network for ucf101 dataset, use pretrained weights on UCF101 for testing
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :param weights: pretrained weights, if None, return network with random initialization
    :return: Output tensor for 101 classes
    """

    # create c3d network with pretrained ucf101 weights
    net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[2]),
                           bias_initializer=tf.constant_initializer(weights[3]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[4]),
                           bias_initializer=tf.constant_initializer(weights[5]))
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[6]),
                           bias_initializer=tf.constant_initializer(weights[7]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[8]),
                           bias_initializer=tf.constant_initializer(weights[9]))
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[10]),
                           bias_initializer=tf.constant_initializer(weights[11]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  # manually padding here :)

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[12]),
                           bias_initializer=tf.constant_initializer(weights[13]))
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  # another manually padding here ;)
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[14]),
                           bias_initializer=tf.constant_initializer(weights[15]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[16]),
                          bias_initializer=tf.constant_initializer(weights[17]))
    net = tf.identity(net, name='fc1')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[18]),
                          bias_initializer=tf.constant_initializer(weights[19]))
    net = tf.identity(net, name='fc2')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=101, activation=None,
                          kernel_initializer=tf.constant_initializer(weights[20]),
                          bias_initializer=tf.constant_initializer(weights[21]))
    net = tf.identity(net, name='logits')

    return net


def c3d_ucf101_finetune(inputs, training, weights=None):
    """
    C3D network for ucf101 dataset fine-tuned from weights pretrained on Sports1M
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :param weights: pretrained weights, if None, return network with random initialization
    :return: Output tensor for 101 classes
    """

    # create c3d network with pretrained Sports1M weights
    net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[2]),
                           bias_initializer=tf.constant_initializer(weights[3]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[4]),
                           bias_initializer=tf.constant_initializer(weights[5]))
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[6]),
                           bias_initializer=tf.constant_initializer(weights[7]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[8]),
                           bias_initializer=tf.constant_initializer(weights[9]))
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[10]),
                           bias_initializer=tf.constant_initializer(weights[11]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  # manually padding here :)

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[12]),
                           bias_initializer=tf.constant_initializer(weights[13]))
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])  # another manually padding here ;)
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[14]),
                           bias_initializer=tf.constant_initializer(weights[15]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[16]),
                          bias_initializer=tf.constant_initializer(weights[17]))
    net = tf.identity(net, name='fc1')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[18]),
                          bias_initializer=tf.constant_initializer(weights[19]))
    net = tf.identity(net, name='fc2')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=101, activation=None,
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                          bias_initializer=tf.zeros_initializer())

    net = tf.identity(net, name='logits')

    return net


def read_train(tr_file):
    path, frm, cls = tr_file.split(' ')
    start = np.random.randint(int(frm) - FRAMES)

    frame_dir = './frames/'

    v_paths = [frame_dir + path + 'frm_%06d.jpg' % (f + 1) for f in range(start, start + FRAMES)]

    offsets = randcrop(scales=[128, 112, 96, 84], size=(128, 171))
    voxel = clipread(v_paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB')

    is_flip = np.random.rand(1, 1).squeeze() > 0.5
    if is_flip:
        voxel = np.flip(voxel, axis=2)

    return voxel, np.float32(cls)


def read_test(tst_file):
    path, start, cls, vid = tst_file.split(' ')
    start = int(start)

    frame_dir = './frames/'
    v_paths = [frame_dir + path + 'frm_%06d.jpg' % (f + 1) for f in range(start, start + FRAMES)]
    offsets = [8, 8 + 112, 30, 30 + 112] # center crop
    voxel = clipread(v_paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB')

    return voxel, np.float32(cls), np.float32(vid)


def demo_test():
    # Demo of testing on UCF101
    # Each line in test/val file is in form: path start_frame class video
    with open('./list/c3d_demo01.txt', 'r') as f:
        lines = f.read().split('\n')
    tst_files = [line for line in lines if len(line) > 0]

    weights = sio.loadmat('pretrained/c3d_ucf101_tf.mat', squeeze_me=True)['weights']

    # Define placeholders
    x = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
    y = tf.placeholder(tf.int64, None, name='input_y')
    training = tf.placeholder(tf.bool, name='training')

    # Define the C3D model for UCF101
    # inputs = x - tf.constant([90.2, 97.6,  101.4], dtype=tf.float32, shape=[1, 1, 1, 1, 3])
    inputs = x - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
    logits = c3d_ucf101(inputs=inputs, training=training, weights=weights)

    correct_opt = tf.equal(tf.argmax(logits, 1), y, name='correct')
    acc_opt = tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        n_sample = len(tst_files)
        for epoch in range(1):

            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

            bidx = 0
            accuracy = []
            scores = []
            
            clss = []
            vids = []
            
            for idx, tst_file in enumerate(tst_files):
                voxel, cls, vid = read_test(tst_file)
                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                
                clss.append(cls)
                vids.append(vid)

                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_sample:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: False}
                    acc, score = sess.run([acc_opt, logits], feed_dict=feeds)

                    scores.append(score)
                    accuracy.append(acc * bidx)

                    print '%05d/%05d' % (idx, n_sample), 'acc: %.2f' % acc, ctime()

                    bidx = 0

            print 'Acc: ', np.sum(accuracy) / n_sample
            mat = dict()
            mat['scores'] = np.vstack(scores).transpose()
            mat['cls'] = np.array(clss)
            mat['vid'] = np.array(vids)

            sio.savemat('./c3d_demo_01.mat', mat)


def demo_finetune():
    # Demo of training on UCF101
    with open('./list/c3d_train01.txt', 'r') as f:
        lines = f.read().split('\n')
    tr_files = [line for line in lines if len(line) > 0]

    weights = sio.loadmat('pretrained/c3d_sports1m_tf.mat', squeeze_me=True)['weights']

    # Define placeholders
    x = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
    y = tf.placeholder(tf.int64, None, name='input_y')
    training = tf.placeholder(tf.bool, name='training')

    # Define the C3D model for UCF101
    inputs = x - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
    logits = c3d_ucf101_finetune(inputs=inputs, training=training, weights=weights)
    labels = tf.one_hot(y, 101, name='labels')

    # Some operations
    correct_opt = tf.equal(tf.argmax(logits, 1), y, name='correct')
    acc_opt = tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')

    # Define training opt
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='loss')

    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=1000,
                                               decay_rate=0.96, staircase=True)
    train_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        n_train = len(tr_files)
        for epoch in range(30):
            tr_files = shuffle(tr_files)
            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

            bidx = 0
            for idx, tr_file in enumerate(tr_files):
                voxel, cls = read_train(tr_file)

                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: True}
                    _, lss, acc = sess.run([train_opt, loss, acc_opt], feed_dict=feeds)

                    print '%04d/%04d/%04d, loss: %.3f, acc: %.2f' % (idx / BATCH_SIZE, idx, n_train, lss, acc), ctime()

                    # reset batch
                    bidx = 0


if __name__ == '__main__':
    demo_test()
    # demo_finetune()
