import tensorflow as tf
import numpy as np
from utils import tf_util
from tf_ops.nn_distance import tf_nndistance

BN_INIT_DECAY = 0.5
BN_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99


def placeholder_inputs(batch_size, num_point):
    point_clouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    gt_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return point_clouds_ph, gt_ph


def build_parts_aes_graphs(num_parts, num_points, part_embedding_dim, base_learning_rate, batch_size, decay_step,
                           decay_rate, bn_decay_step):
    point_clouds_phs = []
    ae_ops = []
    for i in xrange(num_parts):
        with tf.variable_scope('part' + str(i)):
            print 'Graph part ' + str(i)
            point_clouds_ph, gt_ph = placeholder_inputs(batch_size, num_points)
            point_clouds_phs.append(point_clouds_ph)

            is_training_ph = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY, batch * batch_size, bn_decay_step, BN_DECAY_RATE,
                                                     staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
            bn_decay_summary = tf.summary.scalar('bn_decay' + ' ' + str(i), bn_decay)

            print "Get model and loss"
            pred, end_points = get_model_ae(point_clouds_ph, is_training_ph, batch_size, num_points, bn_decay,
                                            part_embedding_dim)
            loss, end_points_tmp = get_loss_ae(pred, gt_ph, end_points)
            loss_summary = tf.summary.scalar('loss', loss)

            print "Get training operator"
            learning_rate = tf.train.exponential_decay(base_learning_rate, batch * batch_size, decay_step,
                                                       decay_rate, staircase=True)
            learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            merged_summary = tf.summary.merge((bn_decay_summary, loss_summary, learning_rate_summary))

            ae_ops.append({'point_clouds_ph': point_clouds_ph,
                           'gt_ph': gt_ph,
                           'is_training_ph': is_training_ph,
                           'pred': pred,
                           'loss': loss,
                           'train_op': train_op,
                           'merged_summary': merged_summary,
                           'step': batch,
                           'end_points': end_points})

    return ae_ops, point_clouds_phs


def get_model_ae(point_cloud, is_training, batch_size, num_point, bn_decay=None, embedding_dim=64, reuse=False):
    input_point_cloud = tf.expand_dims(point_cloud, -1)
    net, end_points = ae_encoder(batch_size, num_point, 3, input_point_cloud, is_training, bn_decay=bn_decay,
                                 embedding_dim=embedding_dim)
    net = ae_decoder(batch_size, num_point, net, is_training, bn_decay=bn_decay, reuse=reuse)

    return net, end_points


def ae_encoder(batch_size, num_point, point_dim, input_image, is_training, bn_decay=None, embedding_dim=128):
    net = tf_util.conv2d(input_image, 64, [1, point_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, embedding_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                     padding='VALID', scope='maxpool')
    net = tf.reshape(global_feat, [batch_size, -1])
    end_points = {'embedding': net}

    return net, end_points


def ae_decoder(batch_size, num_point, net, is_training, bn_decay=None, reuse=False):
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay,
                                  reuse=reuse)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay,
                                  reuse=reuse)
    net = tf_util.fully_connected(net, num_point * 3, activation_fn=None, scope='fc3', reuse=reuse)
    net = tf.reshape(net, (batch_size, num_point, 3))

    return net


def get_loss_ae(pred, gt, end_points):
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, gt)
    loss = tf.reduce_mean(dists_forward + dists_backward)
    end_points['pcloss'] = loss

    loss = loss * 100
    end_points['loss'] = loss
    return loss, end_points


def build_parts_pcn_graph(ae_ops, point_clouds_ph, num_parts, num_points, noise_embedding_dim, base_learning_rate,
                          batch_size, decay_step, decay_rate, bn_decay_step):
    print '\nGraph PCN'
    with tf.variable_scope('pcn'):
        pcn_is_training = tf.placeholder(tf.bool, shape=())
        y = tf.placeholder(tf.float32, [None, num_parts, num_points, 3], name='y')
        y_mask = tf.placeholder(tf.float32, [None, num_parts, num_points], name='y_mask')
        noise = tf.placeholder(tf.float32, shape=[None, noise_embedding_dim])

        pcn_enc = tf.concat([ae_ops[0]['end_points']['embedding'], ae_ops[1]['end_points']['embedding']], axis=-1)
        for p in xrange(2, num_parts):
            pcn_enc = tf.concat([pcn_enc, ae_ops[p]['end_points']['embedding']], axis=-1)
        pcn_enc = tf.concat([pcn_enc, noise], axis=-1)

        pcn_batch = tf.Variable(0)
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY, pcn_batch * batch_size, bn_decay_step, BN_DECAY_RATE,
                                                 staircase=True)
        pcn_bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        bn_decay_summary = tf.summary.scalar('bn_decay', pcn_bn_decay)

        print "--- Get model and loss"
        x = tf.stack(point_clouds_ph, axis=1)
        y_hat = get_model_pcn(pcn_enc, x, num_parts, pcn_is_training, bn_decay=pcn_bn_decay)
        pcn_loss = get_loss_pcn(y, y_hat, y_mask, num_parts, batch_size)
        loss_summary = tf.summary.scalar('loss', pcn_loss)

        print "--- Get training operator"
        pcn_learning_rate = tf.train.exponential_decay(base_learning_rate, pcn_batch * batch_size, decay_step,
                                                       decay_rate, staircase=True)
        learning_rate_summary = tf.summary.scalar('learning_rate', pcn_learning_rate)
        pcn_param = [var for var in tf.trainable_variables() if any(x in var.name for x in ['pcn'])]
        optimizer = tf.train.AdamOptimizer(pcn_learning_rate).minimize(pcn_loss, global_step=pcn_batch,
                                                                       var_list=pcn_param)

        merged_summary = tf.summary.merge((bn_decay_summary, loss_summary, learning_rate_summary))

        pcn_ops = ({'y': y,
                    'y_mask': y_mask,
                    'noise': noise,
                    'is_training_ph': pcn_is_training,
                    'pcn_enc': pcn_enc,
                    'pred': y_hat,
                    'loss': pcn_loss,
                    'train_op': optimizer,
                    'merged_summary': merged_summary,
                    'step': pcn_batch})

        return pcn_ops


def get_model_pcn(pcn_enc, x, num_parts, is_training, bn_decay=None, reuse=False):
    with tf.variable_scope('trans', reuse=reuse):
        bias_initializer = np.array([[1., 0, 1, 0, 1, 0] for _ in xrange(num_parts)])
        bias_initializer = bias_initializer.astype('float32').flatten()

        net = tf_util.fully_connected(pcn_enc, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay,
                                      reuse=reuse)
        net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay,
                                      reuse=reuse)
        trans = tf_util.fully_connected(net, num_parts * 6, activation_fn=None, scope='fc3',
                                        weights_initializer=tf.zeros_initializer(),
                                        biases_initializer=tf.constant_initializer(bias_initializer), reuse=reuse)

    # Perform transformation
    with tf.variable_scope('pcn', reuse=reuse):
        zeros_dims = tf.stack([tf.shape(x)[0], 1])
        zeros_col = tf.fill(zeros_dims, 0.0)
        '''
        sx  0   0   tx
        0   sy  0   ty
        0   0   sz  tz
        '''
        trans_mat = tf.concat((tf.expand_dims(trans[:, 0], axis=1), zeros_col, zeros_col,
                               tf.expand_dims(trans[:, 1], axis=1), zeros_col, tf.expand_dims(trans[:, 2], axis=1),
                               zeros_col, tf.expand_dims(trans[:, 3], axis=1), zeros_col, zeros_col,
                               tf.expand_dims(trans[:, 4], axis=1), tf.expand_dims(trans[:, 5], axis=1),

                               tf.expand_dims(trans[:, 6], axis=1), zeros_col, zeros_col,
                               tf.expand_dims(trans[:, 7], axis=1), zeros_col, tf.expand_dims(trans[:, 8], axis=1),
                               zeros_col, tf.expand_dims(trans[:, 9], axis=1), zeros_col, zeros_col,
                               tf.expand_dims(trans[:, 10], axis=1), tf.expand_dims(trans[:, 11], axis=1)), axis=1)
        for p in xrange(2, num_parts):
            start_ind = 6 * p
            trans_mat = tf.concat((trans_mat, tf.expand_dims(trans[:, start_ind], axis=1), zeros_col, zeros_col,
                                   tf.expand_dims(trans[:, start_ind + 1], axis=1), zeros_col,
                                   tf.expand_dims(trans[:, start_ind + 2], axis=1), zeros_col,
                                   tf.expand_dims(trans[:, start_ind + 3], axis=1), zeros_col, zeros_col,
                                   tf.expand_dims(trans[:, start_ind + 4], axis=1),
                                   tf.expand_dims(trans[:, start_ind + 5], axis=1)), axis=1)

        trans_mat = tf.reshape(trans_mat, (-1, num_parts, 3, 4))
        # adding 1 (w coordinate) to every point (x,y,z,1)
        w = tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])
        x = tf.concat((x, w), axis=-1)
        x_t = tf.transpose(x, [0, 1, 3, 2])
        y_hat_t = tf.matmul(trans_mat, x_t)
        y_hat = tf.transpose(y_hat_t, [0, 1, 3, 2])

    return y_hat


def get_loss_pcn(gt, pred, gt_mask, num_parts, batch_size):
    dists_forward_total = tf.zeros(batch_size)
    dists_backward_total = tf.zeros(batch_size)
    for part in xrange(num_parts):
        dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred[:, part], gt[:, part])
        # zero out the non-existing parts
        dists_forward = tf.reduce_sum(tf.multiply(dists_forward, gt_mask[:, part]), axis=-1)
        dists_backward = tf.reduce_sum(tf.multiply(dists_backward, gt_mask[:, part]), axis=-1)
        dists_forward_total += dists_forward
        dists_backward_total += dists_backward

    loss = dists_forward_total + dists_backward_total
    # divide by the number of parts
    div = tf.reduce_sum(tf.reduce_mean(gt_mask, axis=-1), axis=-1)
    loss = tf.reduce_mean(tf.div(loss, div))

    return loss * 100


def build_test_graph(num_parts, num_points, part_embedding_dim, base_learning_rate, batch_size, decay_step, decay_rate,
                     bn_decay_step, noise_embedding_dim):
    ae_ops, point_clouds_ph = build_parts_aes_graphs(num_parts, num_points, part_embedding_dim, base_learning_rate,
                                                     batch_size, decay_step, decay_rate, bn_decay_step)
    for i in xrange(num_parts):
        with tf.variable_scope('part' + str(i)):
            samples = (tf.placeholder(tf.float32, shape=(batch_size, part_embedding_dim)))
            dec = ae_decoder(batch_size, num_points, samples, ae_ops[i]["is_training_ph"], reuse=True)
            ae_ops[i]['samples'] = samples
            ae_ops[i]['dec'] = dec

    pcn_ops = build_parts_pcn_graph(ae_ops, point_clouds_ph, num_parts, num_points, noise_embedding_dim,
                                    base_learning_rate, batch_size, decay_step, decay_rate, bn_decay_step)

    with tf.variable_scope('pcn'):
        x_full = tf.stack((ae_ops[0]['dec'], ae_ops[1]['dec']), axis=1)
        for p in xrange(2, num_parts):
            x_full = tf.concat((x_full, tf.expand_dims(ae_ops[p]['dec'], axis=1)), axis=1)

        cpcn_enc_full = tf.concat([ae_ops[0]['samples'], ae_ops[1]['samples']], axis=-1)
        for p in xrange(2, num_parts):
            cpcn_enc_full = tf.concat([cpcn_enc_full, ae_ops[p]['samples']], axis=-1)
        cpcn_enc_full = tf.concat([cpcn_enc_full, pcn_ops['noise']], axis=-1)

        y_hat_full = get_model_pcn(cpcn_enc_full, x_full, num_parts, pcn_ops['is_training_ph'], reuse=True)
        pcn_ops['pred_full'] = y_hat_full

    return ae_ops, pcn_ops
