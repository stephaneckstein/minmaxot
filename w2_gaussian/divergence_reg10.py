import numpy as np
import tensorflow as tf
import time
from ff_nets import ff_net


# Specifying problem parameters
BATCH_THETA = 2 ** 12
BATCH_MU = 2 ** 12

N = 20000  # total number of training steps for generator
N_REPORT = 100  # report values each N_REPORT many training steps
N_SAMPS_TOTAL = 20000  # number of samples generated
SAVE_VALUES = 1  # if 0, then no values are saved
N_inf = 10  # number of infimum steps for each generator step; set to either 1 or 10 for values in the paper
N_r = 500
N_s = 200  # warm up phase for generator without infimum steps. Otherwise, particularly if many infimum steps are
# used, the generator sometimes gets stuck at very concentrated and bad measures. Algorithmically, this should be
# regarded as an extended initialization for the weights of the generator.

# Variable parameters
HIDDEN = 64  # 64 or 256
boc = 0  # 0 means base case, 1 means combined (unrolling+mixture) case
D_EACH = 1  # 1, 2, 3, 5, 10, is the dimension of each Gaussian measure in the problem
N_RUNS = 1  # number of identical experiment runs

LAYERS_H = 4
HIDDEN_H = HIDDEN
LAYERS_T = 4
HIDDEN_T = HIDDEN
ACT_T = 'tanh'
ACT_H = 'ReLu'

D = 2*D_EACH
D_THETA = D

# specifying the measures mu, theta, the divergence functions psi_star and the objective function f
def sample_mu(batch_size):
    while 1:
        dataset = np.zeros([batch_size, D])
        dataset[:, 0:D_EACH] = np.random.multivariate_normal(np.zeros([D_EACH]), 1*np.eye(D_EACH), size=[batch_size, D_EACH])[:,:,0]
        dataset[:, D_EACH:D] = np.random.multivariate_normal(np.zeros([D_EACH]), 4*np.eye(D_EACH), size=[batch_size, D_EACH])[:,:,0]
        yield dataset


def sample_theta(batch_size):   # latent measure
    while 1:
        yield np.random.uniform(0, 2, [batch_size, D_THETA])


def psi_star(s):
    return 1/150 * tf.square(s)
    # return 1/4*tf.square(s) +s

def f_objective(u):
    return -tf.reduce_sum(tf.square(u[:, 0:D_EACH]-u[:, D_EACH:D]), axis=1)


# build tensorflow graph
for run_K in range(N_RUNS):
    t0 = time.time()
    tf.reset_default_graph()
    #
    x_mu = tf.placeholder(shape=[None, D], dtype=tf.float32)  # samples from mu
    x_theta = tf.placeholder(shape=[None, D_THETA], dtype=tf.float32)  # samples from theta
    #
    T_theta = ff_net(x_theta, 'T', input_dim=D_THETA, output_dim=D, activation=ACT_T, n_layers=LAYERS_T, hidden_dim=HIDDEN_T)
    #
    h_mu = 0  # sum over h evaluated at samples of mu
    h_T_theta = 0  # sum over h evaluated at samples of theta
    for i in range(2):
        h_mu_h = ff_net(x_mu[:, i*D_EACH:(i + 1)*D_EACH], 'h_' + str(i), input_dim=D_EACH, output_dim=1, activation=ACT_H,
                       n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
        h_mu += h_mu_h # + psi_star(h_mu_h)
        h_T_theta += ff_net(T_theta[:, i*D_EACH:(i + 1)*D_EACH], 'h_' + str(i), input_dim=D_EACH, output_dim=1, activation=ACT_H,
                            n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
    #
    obj = tf.reduce_mean(f_objective(T_theta)) - tf.reduce_mean(h_T_theta) + tf.reduce_mean(h_mu)
    integral = tf.reduce_mean(f_objective(T_theta))
    #
    T_vars = [v for v in tf.compat.v1.global_variables() if ('T' in v.name)]
    h_vars = [v for v in tf.compat.v1.global_variables() if ('h' in v.name)]
    #
    train_op_h = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        obj, var_list=h_vars)
    train_op_T = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9, epsilon=1e-08).minimize(
        -obj, var_list=T_vars)
    #
    # training and saving values
    objective_values = []
    samp_mu = sample_mu(BATCH_MU)
    samp_theta = sample_theta(BATCH_THETA)
    integral_values = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(1, N + 1):
            for _ in range(N_inf):
                if i > N_s:
                    s_mus = next(samp_mu)
                    s_theta = next(samp_theta)
                    (_,) = sess.run([train_op_h], feed_dict={x_mu: s_mus, x_theta: s_theta})
            s_mus = next(samp_mu)
            s_theta = next(samp_theta)
            (_, ov, iv) = sess.run([train_op_T, obj, integral], feed_dict={x_mu: s_mus, x_theta: s_theta})
            integral_values.append(iv)
            objective_values.append(ov)
            if i % N_REPORT == 0:
                print(i, 'objective value = ' + str(np.mean(objective_values[-N_r:])))
                print(i, 'integral value = ' + str(np.mean(integral_values[-N_r:])))
                print('runtime: ' + str(time.time() - t0))
        print('final objective value = ' + str(np.mean(objective_values[-N_r:])))
        print('final integral value = ' + str(np.mean(integral_values[-N_r:])))
        print('total runtime for this run: ' + str(time.time() - t0))
        #
        if SAVE_VALUES == 1:
            np.savetxt('output/objective_values_divergence_' + str(HIDDEN) + '_' + str(
                D_EACH) + '_' + str(run_K), objective_values)
            np.savetxt('output/integral_values_divergence_' + str(HIDDEN) + '_' + str(
                D_EACH) + '_' + str(run_K), integral_values)
