import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import os
import sys
import models
from data import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='datasets/shapenetcore_partanno_segmentation_benchmark_v0',
                    help='Path to the dataset [default: datasets/shapenetcore_partanno_segmentation_benchmark_v0]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default='Chair', help='Which single class to train on [default: Chair]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--part_embedding_dim', type=int, default=64, help='Embedding dimension of each part [default: 64]')
parser.add_argument('--noise_embedding_dim', type=int, default=16,
                    help='Embedding dimension of the noise [default: 16]')
parser.add_argument('--num_point', type=int, default=400, help='Number of points per part [default: 400]')
parser.add_argument('--max_epoch_ae', type=int, default=401, help='Number of epochs for each AEs [default: 401]')
parser.add_argument('--max_epoch_pcn', type=int, default=201,
                    help='Number of epochs for the parts composition network [default: 201]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate for the composition network [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()
GPU_INDEX = FLAGS.gpu
CATEGORY = FLAGS.category
LOG_DIR = FLAGS.log_dir
PART_EMBEDDING_DIM = FLAGS.part_embedding_dim
NOISE_EMBEDDING_DIM = FLAGS.noise_embedding_dim
NUM_POINTS = FLAGS.num_point
MAX_EPOCH_AE = FLAGS.max_epoch_ae
MAX_EPOCH_PCN = FLAGS.max_epoch_pcn
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

print 'Loading data'
# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, FLAGS.data_path)
# Using the smaller set, test, as seen and the larger, trainval, as unseen
PCN_TRAIN_DATASET, PCN_TEST_DATASET, AE_TRAIN_DATASET, AE_TEST_DATASET, NUM_PARTS = data_utils.load_data(DATA_PATH,
                                                                                                         NUM_POINTS,
                                                                                                         CATEGORY,
                                                                                                         'test',
                                                                                                         'trainval')

NOISE = np.random.normal(size=[len(PCN_TRAIN_DATASET), NOISE_EMBEDDING_DIM])
EPOCH_CNT = 0


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_ae_batch(dataset, idxs, start_idx, end_idx):
    batch_size = end_idx - start_idx
    batch_data = np.zeros((batch_size, NUM_POINTS, 3))
    for i in range(batch_size):
        ps = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
    return batch_data


def train_ae_one_epoch(sess, ops_ae, train_dataset, train_writer):
    is_training = True
    log_string(str(datetime.now()))

    # Shuffle train samples
    train_idxs = np.arange(0, len(train_dataset))
    np.random.shuffle(train_idxs)
    num_batches = len(train_dataset) / BATCH_SIZE

    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data = get_ae_batch(train_dataset, train_idxs, start_idx, end_idx)
        feed_dict = {ops_ae['point_clouds_ph']: batch_data,
                     ops_ae['gt_ph']: batch_data,
                     ops_ae['is_training_ph']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops_ae['merged_summary'], ops_ae['step'],
                                                         ops_ae['train_op'], ops_ae['loss'],
                                                         ops_ae['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx + 1) % 10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0


def get_pcn_batch(dataset, idxs, start_idx, end_idx):
    batch_size = end_idx - start_idx
    x = np.zeros((NUM_PARTS, batch_size, NUM_POINTS, 3))
    y = np.zeros((batch_size, NUM_PARTS, NUM_POINTS, 3))
    y_mask = np.zeros((batch_size, NUM_PARTS, NUM_POINTS))
    noise = NOISE[idxs[start_idx:end_idx]]

    for i in range(batch_size):
        point_sets = dataset[idxs[start_idx + i]]
        for p in xrange(NUM_PARTS):
            ps, sn, is_full = point_sets[p]
            x[p, i, ...] = sn
            y[i, p, ...] = ps
            y_mask[i, p] = is_full
    return x, y, y_mask, noise


def train_pcn_one_epoch(sess, pcn_ops, ae_ops, train_dataset, train_writer):
    is_training = True
    log_string(str(datetime.now()))

    # Shuffle train samples
    train_idxs = np.arange(0, len(train_dataset))
    np.random.shuffle(train_idxs)
    num_batches = len(train_dataset) / BATCH_SIZE

    loss_sum = 0
    total_loss = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        x, y, y_mask, noise = get_pcn_batch(train_dataset, train_idxs, start_idx, end_idx)
        feed_dict = {pcn_ops['y']: y,
                     pcn_ops['y_mask']: y_mask,
                     pcn_ops['noise']: noise,
                     pcn_ops['is_training_ph']: is_training, }

        for part in xrange(len(x)):
            feed_dict[ae_ops[part]['point_clouds_ph']] = x[part]
            feed_dict[ae_ops[part]['gt_ph']] = x[part]
            feed_dict[ae_ops[part]['is_training_ph']] = False  # encoders are set in the PCN training phase

        summary, step, _, loss_val, pred_val = sess.run(
            [pcn_ops['merged_summary'], pcn_ops['step'], pcn_ops['train_op'], pcn_ops['loss'], pcn_ops['pred']],
            feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        loss_sum += loss_val
        total_loss += loss_val

        if (batch_idx + 1) % 10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0

    return total_loss / float(num_batches)


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            ''' Parts' AE '''
            ae_ops, point_clouds_ph = models.build_parts_aes_graphs(NUM_PARTS, NUM_POINTS, PART_EMBEDDING_DIM,
                                                                    BASE_LEARNING_RATE, BATCH_SIZE, DECAY_STEP,
                                                                    DECAY_RATE, float(DECAY_STEP))

            ''' PCN '''
            pcn_ops = models.build_parts_pcn_graph(ae_ops, point_clouds_ph, NUM_PARTS, NUM_POINTS, NOISE_EMBEDDING_DIM,
                                                   BASE_LEARNING_RATE, BATCH_SIZE, DECAY_STEP, DECAY_RATE,
                                                   float(DECAY_STEP))

            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        ''' Train Parts AE '''
        for i in xrange(NUM_PARTS):
            print 'Training part ' + str(i)
            for epoch in range(MAX_EPOCH_AE):
                log_string('**** EPOCH %03d ****' % epoch)
                sys.stdout.flush()
                train_ae_one_epoch(sess, ae_ops[i], AE_TRAIN_DATASET[i], train_writer)

        ''' Train PCN '''
        best_loss = 1e20
        for epoch in range(MAX_EPOCH_PCN):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()
            train_loss = train_pcn_one_epoch(sess, pcn_ops, ae_ops, PCN_TRAIN_DATASET, train_writer)
            if train_loss < best_loss:
                best_loss = train_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % epoch))
                log_string("Model saved in file: %s" % save_path)
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
