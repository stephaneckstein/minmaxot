import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
from ff_nets import ff_net
import time

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

# mixture parameters:
N_GEN = 5  # number of mixture components for the generator
A_GEN = np.append(np.array([0]), np.cumsum(1 / N_GEN * np.ones(N_GEN)))  # mixture is equally weighted

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
        yield np.random.uniform(-1, 1, [batch_size, 2, N_GEN])


def inds_generator(batch_size):
    while 1:
        data = np.zeros([batch_size, 2, N_GEN])
        w = np.random.random_sample(batch_size)
        for i in range(N_GEN):
            w_i = ((A_GEN[i] <= w).astype(float)) * (w < A_GEN[i + 1]).astype(float)
            for j in range(2):
                data[:, j, i] = w_i
        yield data


def f_objective(u):
    return tf.nn.relu(u[:, 1] - u[:, 0])


# build tensorflow graph
for run_K in range(N_RUNS):
    t0 = time.time()
    tf.reset_default_graph()

    x_mu = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    x_theta = tf.placeholder(shape=[None, 2, N_GEN], dtype=tf.float32)
    inds_gen = tf.placeholder(shape=[None, 2, N_GEN],dtype=tf.float32)  # determines which generator is chosen for each
                                                                        # of the samples.
                                                                        # should be constant along the second dimension,
                                                                        # with entries either 0 or 1 so that
                                                                        # inds[i, j, :] sums to one
    y = 0
    for i in range(N_GEN):
        y += inds_gen[:, :, i] * ff_net(x_theta[:, :, i], 'T_'+str(i), input_dim=2, output_dim=2, activation=ACT_T,
                                    n_layers=LAYERS_T, hidden_dim=HIDDEN_T)
    T_theta = y

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

    train_op_h = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        obj, var_list=h_vars)
    train_op_T = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        -obj, var_list=T_vars)

    # training and saving values
    integral_values = []
    objective_values = []
    samp_mu = sample_mu(BATCH_MU)
    samp_theta = sample_theta(BATCH_THETA)
    samp_ind_gen = inds_generator(BATCH_THETA)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(1, N + 1):
            for _ in range(N_inf):
                s_mus = next(samp_mu)
                s_theta = next(samp_theta)
                s_ind_gen = next(samp_ind_gen)
                (_, ov, iv) = sess.run([train_op_h, obj, integral], feed_dict={x_mu: s_mus, x_theta: s_theta,
                                                                               inds_gen: s_ind_gen})
            s_mus = next(samp_mu)
            s_theta = next(samp_theta)
            s_ind_gen = next(samp_ind_gen)
            (_, ov, iv) = sess.run([train_op_T, obj, integral], feed_dict={x_mu: s_mus, x_theta: s_theta,
                                                                           inds_gen: s_ind_gen})
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
            np.savetxt('output/objective_values_mixture' + str(run_K), objective_values)
            np.savetxt('output/integral_values_mixture' + str(run_K), integral_values)

        if SAVE_SAMPLES == 1:
            all_samples = np.zeros([0, 2])
            while len(all_samples) < N_SAMPS_TOTAL:
                s_theta = next(samp_theta)
                s_ind_gen = next(samp_ind_gen)
                xx = sess.run(T_theta, feed_dict={x_theta: s_theta, inds_gen: s_ind_gen})
                all_samples = np.append(all_samples, xx, axis=0)
            np.savetxt('output/samples_mixture' + str(run_K), all_samples[:N_SAMPS_TOTAL, :])
