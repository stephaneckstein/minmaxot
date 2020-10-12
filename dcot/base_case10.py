import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
from ff_nets import ff_net
import time

# Specifying problem parameters
N_RUNS = 10  # number of identical experiment runs
BATCH_THETA = 2 ** 7
BATCH_MU = 2 ** 7

N = 20000  # total number of training steps for generator
N_REPORT = 100
N_SAMPS_TOTAL = 20000  # number of samples generated
SAVE_VALUES = 1  # if 0, then no values are saved
N_inf = 10  # number of infimum steps for each generator step; set to either 1 or 10 for values in the paper
N_r = 1000
N_s = 200  # warm up phase for generator without infimum steps. Otherwise, particularly if many infimum steps are
# used, the generator sometimes gets stuck at very concentrated and bad measures. Algorithmically, this should be
# regarded as an extended initialization for the weights of the generator.

# network architecture
LAYERS_H = 4
HIDDEN_H = 64
LAYERS_T = 4
HIDDEN_T = 64
ACT_T = 'tanh'
ACT_H = 'ReLu'


# specifying the measures mu, theta and kappa, and the objective function f
def sample_mu(batch_size):
    while 1:
        dataset = np.random.normal(0, 2, [batch_size, 2])
        yield dataset


def sample_kappa(batch_size):
    while 1:
        yield np.random.standard_t(8, [batch_size, 1])


def sample_theta(batch_size):
    while 1:
        yield np.random.uniform(-1, 1, [batch_size, 2])


def f_objective(u):
    return tf.nn.relu(tf.reduce_sum(u, axis=1))


# build tensorflow graph
for run_K in range(N_RUNS):
    t0 = time.time()
    tf.reset_default_graph()

    x_mu = tf.placeholder(shape=[None, 2], dtype=tf.float32)  # samples from mu
    x_theta = tf.placeholder(shape=[None, 2], dtype=tf.float32)  # samples from theta
    x_kappa = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # samples from kappa

    T_theta = ff_net(x_theta, 'T', input_dim=2, output_dim=2, activation=ACT_T, n_layers=LAYERS_T, hidden_dim=HIDDEN_T)

    h_mu = 0  # sum over h evaluated at samples of mu
    h_T_theta = 0  # sum over h evaluated at samples of theta
    for i in range(2):
        h_mu += ff_net(x_mu[:, i:(i + 1)], 'h_' + str(i), input_dim=1, output_dim=1, activation=ACT_H,
                       n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
        h_T_theta += ff_net(T_theta[:, i:(i + 1)], 'h_' + str(i), input_dim=1, output_dim=1, activation=ACT_H,
                            n_layers=LAYERS_H, hidden_dim=HIDDEN_H)

    h_kappa_mu = ff_net(x_kappa, 'h_kappa', input_dim=1, output_dim=1, activation=ACT_H, n_layers=LAYERS_H,
                        hidden_dim=HIDDEN_H)
    h_kappa_T_theta = ff_net(T_theta[:, 0:1] - T_theta[:, 1:2], 'h_kappa', input_dim=1, output_dim=1,
                             activation=ACT_H, n_layers=LAYERS_H, hidden_dim=HIDDEN_H)

    obj = tf.reduce_mean(f_objective(T_theta)) - tf.reduce_mean(h_T_theta) + tf.reduce_mean(h_mu) - tf.reduce_mean(
        h_kappa_T_theta) + tf.reduce_mean(h_kappa_mu)

    T_vars = [v for v in tf.compat.v1.global_variables() if ('T' in v.name)]
    h_vars = [v for v in tf.compat.v1.global_variables() if ('h' in v.name)]

    train_op_h = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        obj, var_list=h_vars)
    train_op_T = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        -obj, var_list=T_vars)

    # training and saving values
    objective_values = []
    samp_mu = sample_mu(BATCH_MU)
    samp_theta = sample_theta(BATCH_THETA)
    samp_kappa = sample_kappa(BATCH_MU)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(1, N + 1):
            for _ in range(N_inf):
                if i > N_s:
                    s_mus = next(samp_mu)
                    s_theta = next(samp_theta)
                    s_kappa = next(samp_kappa)
                    (_,) = sess.run([train_op_h], feed_dict={x_mu: s_mus, x_theta: s_theta, x_kappa: s_kappa})
            s_mus = next(samp_mu)
            s_theta = next(samp_theta)
            s_kappa = next(samp_kappa)
            (_, ov) = sess.run([train_op_T, obj], feed_dict={x_mu: s_mus, x_theta: s_theta, x_kappa: s_kappa})
            objective_values.append(ov)
            if i % N_REPORT == 0:
                print(i, 'objective value = ' + str(np.mean(objective_values[-N_r:])))
                print('runtime: ' + str(time.time() - t0))
        print('final objective value = ' + str(np.mean(objective_values[-N_r:])))
        print('total runtime for this run: ' + str(time.time() - t0))

        if SAVE_VALUES == 1:
            np.savetxt('output/objective_values_base' + str(N_inf) + '_' + str(run_K), objective_values)
