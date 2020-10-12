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

LIP_CONST = 1
LIP_LAM = 10

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

    if LIP_CONST > 0:
        x_mu_1 = x_mu[:, 0:1]
        gradients_mu1 = tf.gradients(ff_net(x_mu_1, 'h_0', n_layers=LAYERS_H, hidden_dim=HIDDEN_H, input_dim=1,
                                            output_dim=1, activation=ACT_H), [x_mu_1])[0]
        x_mu_2 = x_mu[:, 1:2]
        gradients_mu2 = tf.gradients(ff_net(x_mu_2, 'h_1', n_layers=LAYERS_H, hidden_dim=HIDDEN_H, input_dim=1,
                                            output_dim=1, activation=ACT_H), [x_mu_2])[0]
        gradients_kappa_mu = tf.gradients(ff_net(x_kappa, 'h_kappa', n_layers=LAYERS_H, hidden_dim=HIDDEN_H, input_dim=1,
                                            output_dim=1, activation=ACT_H), [x_kappa])[0]

        x_T_1 = T_theta[:, 0:1]
        gradients_T1 = tf.gradients(ff_net(x_T_1, 'h_0', n_layers=LAYERS_H, hidden_dim=HIDDEN_H, input_dim=1,
                                            output_dim=1, activation=ACT_H), [x_T_1])[0]
        x_T_2 = T_theta[:, 1:2]
        gradients_T2 = tf.gradients(ff_net(x_T_2, 'h_1', n_layers=LAYERS_H, hidden_dim=HIDDEN_H, input_dim=1,
                                            output_dim=1, activation=ACT_H), [x_T_2])[0]
        x_kappa_T = T_theta[:, 0:1] - T_theta[:, 1:2]
        gradients_kappa_T = tf.gradients(ff_net(x_kappa_T, 'h_kappa', n_layers=LAYERS_H, hidden_dim=HIDDEN_H, input_dim=1,
                                            output_dim=1, activation=ACT_H), [x_kappa_T])[0]

        sqrt_eps = 10 ** (-9)  # this is added to avoid infinite gradients of the square root at 0,
        # which can otherwise sometimes occur late during training

        slopes_mu1 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(gradients_mu1), reduction_indices=[1]))
        slopes_mu2 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(gradients_mu2), reduction_indices=[1]))
        slopes_kappa_mu = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(gradients_kappa_mu), reduction_indices=[1]))

        slopes_T1 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(gradients_T1), reduction_indices=[1]))
        slopes_T2 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(gradients_T2), reduction_indices=[1]))
        slopes_kappa_T = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(gradients_kappa_T), reduction_indices=[1]))

        gpm1 = tf.reduce_mean(tf.nn.relu(slopes_mu1 - LIP_CONST) ** 2)
        gpm2 = tf.reduce_mean(tf.nn.relu(slopes_mu2 - LIP_CONST) ** 2)
        gpmk = tf.reduce_mean(tf.nn.relu(slopes_kappa_mu - LIP_CONST) ** 2)

        gpT1 = tf.reduce_mean(tf.nn.relu(slopes_T1 - LIP_CONST) ** 2)
        gpT2 = tf.reduce_mean(tf.nn.relu(slopes_T2 - LIP_CONST) ** 2)
        gpTk = tf.reduce_mean(tf.nn.relu(slopes_kappa_T - LIP_CONST) ** 2)

        obj_h = obj + LIP_LAM * (gpm1 + gpm2 + gpmk + gpT1 + gpT2 + gpTk)
    else:
        obj_h = obj

    T_vars = [v for v in tf.compat.v1.global_variables() if ('T' in v.name)]
    h_vars = [v for v in tf.compat.v1.global_variables() if ('h' in v.name)]

    train_op_h = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        obj_h, var_list=h_vars)
    train_op_T = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        -obj, var_list=T_vars)

    # training and saving values
    objective_values = []
    samp_mu = sample_mu(BATCH_MU)
    samp_theta = sample_theta(BATCH_THETA)
    samp_kappa = sample_kappa(BATCH_MU)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(1, N + 1):
            if i > N_s:
                for _ in range(N_inf):
                    s_mus = next(samp_mu)
                    s_theta = next(samp_theta)
                    s_kappa = next(samp_kappa)
                    (_,) = sess.run([train_op_h], feed_dict={x_mu: s_mus, x_theta: s_theta, x_kappa: s_kappa})
            s_mus = next(samp_mu)
            s_theta = next(samp_theta)
            s_kappa = next(samp_kappa)
            (_, ov) = sess.run([train_op_T, obj_h], feed_dict={x_mu: s_mus, x_theta: s_theta, x_kappa: s_kappa})
            objective_values.append(ov)
            if i % N_REPORT == 0:
                print(i, 'objective value = ' + str(np.mean(objective_values[-N_r:])))
                print('runtime: ' + str(time.time() - t0))
        print('final objective value = ' + str(np.mean(objective_values[-N_r:])))
        print('total runtime for this run: ' + str(time.time() - t0))

        if SAVE_VALUES == 1:
            np.savetxt('output/objective_values_lipschitz' + str(N_inf) + '_' + str(run_K), objective_values)
