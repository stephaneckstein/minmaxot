import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
from ff_nets import ff_net
import time
from collections import OrderedDict
from keras.optimizers import Adam

# Specifying problem parameters
N_RUNS = 10  # number of identical experiment runs

BATCH_THETA = 2 ** 11
BATCH_MU = 2 ** 11

N = 15000  # total number of training steps for generator
N_REPORT = 100
N_SAMPS_TOTAL = 100000  # number of samples generated
SAVE_SAMPLES = 1  # if 0, then no samples are saved
SAVE_VALUES = 1  # if 0, then no values are saved
N_inf = 1  # number of infimum steps for each generator step
N_r = 1000

# network architecture
LAYERS_H = 4
HIDDEN_H = 128
LAYERS_T = 4
HIDDEN_T = 128
ACT_T = 'tanh'
ACT_H = 'ReLu'

# unrolling parameters:
UNROLLING_STEPS = 5

# specifying the measures mu and theta and the objective function f
mean1 = -1.3
mean2 = 0.8
sig1 = 0.5
sig2 = 0.7
sig3 = 1.1
sig4 = 1.3
mixt = 0.5


def sample_mu(K):
    while 1:
        points1 = np.random.randn(K)
        id = np.random.binomial(1, mixt, size=K)
        points1 = points1 * (id * sig1 + (1 - id) * sig2) + id * mean1 + (1 - id) * mean2
        er2 = np.random.randn(K)
        points2 = points1 + er2 * ((sig3 - sig1) * id + (sig4 - sig2) * (1 - id))
        out = np.zeros([K, 2])
        out[:, 0] = points1
        out[:, 1] = points2
        yield out


def sample_theta(batch_size):
    while 1:
        yield np.random.uniform(-1, 1, [batch_size, 2])


def f_objective(u):
    return tf.nn.relu(u[:, 1] - u[:, 0])


# build tensorflow graph
for run_K in range(N_RUNS):
    t0 = time.time()
    tf.reset_default_graph()

    # functions for unrolling
    def remove_original_op_attributes(graph):
        # Remove _original_op attribute from all operations in a graph
        for op in graph.get_operations():
            op._original_op = None


    _graph_replace = tf.contrib.graph_editor.graph_replace


    def graph_replace(*args, **kwargs):
        # Monkey patch graph_replace so that it works with TF 1.0
        remove_original_op_attributes(tf.get_default_graph())
        return _graph_replace(*args, **kwargs)


    def extract_update_dict(update_ops):
        # Extract variables and their new values from Assign and AssignAdd ops
        # Args:
        #     update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()
        # Returns:
        #     dict mapping from variable values to their updated value
        name_to_var = {v.name: v for v in tf.global_variables()}
        updates = OrderedDict()
        for update in update_ops:
            var_name = update.op.inputs[0].name
            var = name_to_var[var_name]
            value = update.op.inputs[1]
            if update.op.type == 'AssignVariableOp' or update.op.type == 'Assign':
                updates[var.value()] = value
            elif update.op.type == 'AssignAddVariableOp' or update.op.type == 'AssignAdd':
                updates[var.value()] = var + value
            else:
                raise ValueError(
                    "Update op type (%s) must be of type AssignVariableOp, Assign, AssignAdd or AssignAddVariableOp" % update.op.type)
        return updates


    x_mu = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    x_theta = tf.placeholder(shape=[None, 2], dtype=tf.float32)

    T_theta = ff_net(x_theta, 'T', input_dim=2, output_dim=2, activation=ACT_T, n_layers=LAYERS_T, hidden_dim=HIDDEN_T)

    h_mu = 0  # sum over h evaluated at samples of mu
    h_T_theta = 0  # sum over h evaluated at samples of theta
    for i in range(2):
        h_mu += ff_net(x_mu[:, i:(i + 1)], 'h' + str(i), input_dim=1, output_dim=1, activation=ACT_H,
                       n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
        h_T_theta += ff_net(T_theta[:, i:(i + 1)], 'h' + str(i), input_dim=1, output_dim=1, activation=ACT_H,
                            n_layers=LAYERS_H, hidden_dim=HIDDEN_H)

    h_T_mart = ff_net(T_theta[:, 0:1], 'h_mart', input_dim=1, output_dim=1, activation=ACT_H, n_layers=LAYERS_H,
                      hidden_dim=HIDDEN_H)
    h_T_mart = h_T_mart * (T_theta[:, 1:2] - T_theta[:, 0:1])

    h_T_theta = tf.reduce_sum(h_T_theta, axis=1) + tf.reduce_sum(h_T_mart, axis=1)

    obj = tf.reduce_mean(f_objective(T_theta)) + tf.reduce_mean(h_T_theta) - tf.reduce_mean(h_mu)
    integral = tf.reduce_mean(f_objective(T_theta))

    T_vars = [v for v in tf.compat.v1.global_variables() if ('T' in v.name)]
    h_vars = [v for v in tf.compat.v1.global_variables() if ('h' in v.name)]

    d_opt = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    # d_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999, epsilon=1e-8)
    updates = d_opt.get_updates(obj, h_vars)
    train_op_h = tf.group(*updates, name="train_op_h")
    if UNROLLING_STEPS > 0:
        # Get dictionary mapping from variables to their update value after one optimization step
        update_dict = extract_update_dict(updates)
        cur_update_dict = update_dict
        for i in range(UNROLLING_STEPS - 1):
            # Compute variable updates given the previous iteration's updated variable
            cur_update_dict = graph_replace(update_dict, cur_update_dict)
        # Final unrolled loss uses the parameters at the last time step
        unrolled_loss = graph_replace(obj, cur_update_dict)
    else:
        unrolled_loss = obj

    train_op_T = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999, epsilon=1e-08).minimize(
        -unrolled_loss, var_list=T_vars)

    # training and saving values
    integral_values = []
    objective_values = []
    samp_mu = sample_mu(BATCH_MU)
    samp_theta = sample_theta(BATCH_THETA)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(1, N + 1):
            for _ in range(N_inf):
                s_mus = next(samp_mu)
                s_theta = next(samp_theta)
                (_, ov, iv) = sess.run([train_op_h, obj, integral], feed_dict={x_mu: s_mus, x_theta: s_theta})
            s_mus = next(samp_mu)
            s_theta = next(samp_theta)
            (_, ov, iv) = sess.run([train_op_T, obj, integral], feed_dict={x_mu: s_mus, x_theta: s_theta})
            objective_values.append(ov)
            integral_values.append(iv)
            if i % N_REPORT == 0:
                print(i, 'objective value = ' + str(np.mean(objective_values[-N_r:])), 'integral value = ' + str(
                    np.mean(integral_values[-N_r:])))
                print('runtime: ' + str(time.time() - t0))
        print('final objective value = ' + str(np.mean(objective_values[-N_r:])), 'final integral value = ' + str(
            np.mean(integral_values[-N_r:])))
        print('total runtime for this run: ' + str(time.time() - t0))

        if SAVE_VALUES == 1:
            np.savetxt('output/objective_values_unrolling' + str(run_K), objective_values)
            np.savetxt('output/integral_values_unrolling' + str(run_K), integral_values)

        if SAVE_SAMPLES == 1:
            all_samples = np.zeros([0, 2])
            while len(all_samples) < N_SAMPS_TOTAL:
                s_theta = next(samp_theta)
                xx = sess.run(T_theta, feed_dict={x_theta: s_theta})
                all_samples = np.append(all_samples, xx, axis=0)
            np.savetxt('output/samples_unrolling' + str(run_K), all_samples[:N_SAMPS_TOTAL, :])
