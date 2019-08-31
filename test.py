import argparse
import numpy as np
import tensorflow as tf
import os
import models
from data import data_utils
from utils import show3d_balls
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='datasets/shapenetcore_partanno_segmentation_benchmark_v0',
                    help='Path to the dataset [default: datasets/shapenetcore_partanno_segmentation_benchmark_v0]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default='Chair', help='Which single class to train on [default: Chair]')
parser.add_argument('--part_embedding_dim', type=int, default=64, help='Embedding dimension of each part [default: 64]')
parser.add_argument('--noise_embedding_dim', type=int, default=16,
                    help='Embedding dimension of the noise [default: 16]')
parser.add_argument('--num_point', type=int, default=400, help='Number of points per part [default: 400]')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate for the parts composition network [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_parts', type=int, default=0,
                    help='Number of Parts, if set to 0 it will take longer to compute [default: 0]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--num_samples', type=int, default='100',
                    help='Number of generated shapes_embedd to be shown [default: 0]')

FLAGS = parser.parse_args()
GPU_INDEX = FLAGS.gpu
CATEGORY = FLAGS.category
PART_EMBEDDING_DIM = FLAGS.part_embedding_dim
NOISE_EMBEDDING_DIM = FLAGS.noise_embedding_dim
NUM_POINTS = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_PARTS = FLAGS.num_parts
MODEL_PATH = FLAGS.model_path
NUM_SAMPLES = FLAGS.num_samples

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = [np.array([163, 254, 170]), np.array([206, 178, 254]), np.array([248, 250, 132]), np.array([237, 186, 145]),
          np.array([192, 144, 145]), np.array([158, 218, 73])]

print 'Loading data'
# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, FLAGS.data_path)
# Using the same splits as when training
if NUM_PARTS == 0:
    _, _, AE_TRAIN_DATASET, _, NUM_PARTS = data_utils.load_data(DATA_PATH, NUM_POINTS, CATEGORY, 'test', 'trainval')
else:
    AE_TRAIN_DATASET, _ = data_utils.load_aes_data(DATA_PATH, NUM_POINTS, CATEGORY, 'test', 'trainval', NUM_PARTS)


def get_model(batch_size):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            ae_ops, pcn_ops = models.build_test_graph(NUM_PARTS, NUM_POINTS, PART_EMBEDDING_DIM, BASE_LEARNING_RATE,
                                                      batch_size, DECAY_STEP, DECAY_RATE, float(DECAY_STEP),
                                                      NOISE_EMBEDDING_DIM)

        saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)

        return sess, ae_ops, pcn_ops


def compute_embedding_gmm_and_sample_vectors(sess, ae_ops):
    num_gmm_components = 20
    samples = []
    for p in xrange(NUM_PARTS):
        part = []
        print 'Embedding part ' + str(p)
        for i in xrange(len(AE_TRAIN_DATASET[p])):
            ps = AE_TRAIN_DATASET[p][i]
            feed_dict = {ae_ops[p]['point_clouds_ph']: np.expand_dims(ps, axis=0),
                         ae_ops[p]['gt_ph']: np.expand_dims(ps, axis=0),
                         ae_ops[p]['is_training_ph']: False, }

            part.append(sess.run(ae_ops[p]['end_points']['embedding'], feed_dict=feed_dict))
        print 'Compute GMM for part', p
        gmm = GaussianMixture(n_components=num_gmm_components, covariance_type='full')
        gmm.fit(np.squeeze(np.array(part)))
        sample, _ = gmm.sample(n_samples=NUM_SAMPLES)
        sample = sample[np.random.permutation(np.arange(NUM_SAMPLES))]
        samples.append(sample)

    return samples


def generate_shapse_from_vectors(sess, ae_ops, pcn_ops, samples):
    for i in range(NUM_SAMPLES):
        noise = np.random.normal(size=[1, NOISE_EMBEDDING_DIM])
        shapes_embedd = np.stack((np.expand_dims(samples[0][i], axis=0), np.expand_dims(samples[1][i], axis=0)), axis=0)
        for p in xrange(2, NUM_PARTS):
            shapes_embedd = np.concatenate(
                (shapes_embedd, np.expand_dims(np.expand_dims(samples[p][i], axis=0), axis=0)), axis=0)

        # Demonstrate a missing part
        if np.random.randint(1000) % 2 == 0:
            missing_part = True
            shapes_embedd[-1] = np.zeros((1, PART_EMBEDDING_DIM))
        else:
            missing_part = False

        feed_dict = {}
        for p in xrange(NUM_PARTS):
            feed_dict[ae_ops[p]['samples']] = shapes_embedd[p]
            feed_dict[ae_ops[p]['is_training_ph']] = False
        feed_dict[pcn_ops['noise']] = noise
        feed_dict[pcn_ops['is_training_ph']] = False
        pred = sess.run(pcn_ops['pred_full'], feed_dict=feed_dict)

        preds = np.concatenate((pred[0, 0], pred[0, 1]), axis=0)
        for p in xrange(2, NUM_PARTS):
            preds = np.concatenate((preds, pred[0, p]), axis=0)

        show_3d_point_clouds(preds, missing_part)


def show_3d_point_clouds(shapes, is_missing_part):
    colors = np.zeros_like(shapes)
    for p in xrange(NUM_PARTS):
        colors[NUM_POINTS * p:NUM_POINTS * (p + 1), :] = COLORS[p]

    # fix orientation
    shapes[:, 1] *= -1
    shapes = shapes[:, [1, 2, 0]]

    if is_missing_part:
        shapes = shapes[:NUM_POINTS * (NUM_PARTS - 1)]
        colors = colors[:NUM_POINTS * (NUM_PARTS - 1)]
    show3d_balls.showpoints(shapes, c_gt=colors, ballradius=8, normalizecolor=False, background=[255, 255, 255])


def test():
    sess, ae_ops, pcn_ops = get_model(batch_size=1)
    samples = compute_embedding_gmm_and_sample_vectors(sess, ae_ops)
    generate_shapse_from_vectors(sess, ae_ops, pcn_ops, samples)


if __name__ == "__main__":
    test()
