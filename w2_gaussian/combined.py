import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import time
from collections import OrderedDict
from ff_nets import ff_net

# Specifying problem parameters
BATCH_THETA = 2 ** 12
BATCH_MU = 2 ** 12

N = 20000  # total number of training steps for generator
N_REPORT = 100  # report values each N_REPORT many training steps
N_SAMPS_TOTAL = 100000  # number of samples generated
SAVE_SAMPLES = 0  # if 0, then no samples are saved
SAVE_VALUES = 1  # if 0, then no values are saved
N_inf = 1  # number of infimum steps for each generator step
N_r = 500

# Variable parameters
HIDDEN = 64  # 64 or 256
boc = 0  # 0 means base case, 1 means combined (unrolling+mixture) case
D_EACH = 1  # 1, 2, 3, 5, 10, is the dimension of each Gaussian measure in the problem
N_RUNS = 1  # number of identical experiment runs

# network architecture
LAYERS_H = 4
HIDDEN_H = HIDDEN
LAYERS_T = 4
HIDDEN_T = HIDDEN
ACT_T = 'tanh'
ACT_H = 'ReLu'

# mixture and unrolling parameters:
if boc == 0:
    N_GEN = 1  # number of mixture components for the generator
    A_GEN = np.append(np.array([0]), np.cumsum(1 / N_GEN * np.ones(N_GEN)))  # mixture is equally weighted
    UNROLLING_STEPS = 0
else:
    N_GEN = 5  # number of mixture components for the generator
    A_GEN = np.append(np.array([0]), np.cumsum(1 / N_GEN * np.ones(N_GEN)))  # mixture is equally weighted
    UNROLLING_STEPS = 5

# specifying the measures mu and theta and the objective function f
D = 2*D_EACH
D_THETA = D


def sample_mu(batch_size):
    while 1:
        dataset = np.zeros([batch_size, D])
        dataset[:, 0:D_EACH] = np.random.multivariate_normal(np.zeros([D_EACH]), 1*np.eye(D_EACH), size=[batch_size, D_EACH])[:,:,0]
        dataset[:, D_EACH:D] = np.random.multivariate_normal(np.zeros([D_EACH]), 4*np.eye(D_EACH), size=[batch_size, D_EACH])[:,:,0]
        yield dataset


def sample_theta(batch_size):   # latent measure
    while 1:
        yield np.random.uniform(0, 2, [batch_size, D_THETA, N_GEN])


def inds_generator(batch_size):
    while 1:
        data = np.zeros([batch_size, D, N_GEN])
        w = np.random.random_sample(batch_size)
        for i in range(N_GEN):
            w_i = ((A_GEN[i] <= w).astype(float)) * (w < A_GEN[i + 1]).astype(float)
            for j in range(D):
                data[:, j, i] = w_i
        yield data


def f_objective(u):
    return -tf.reduce_sum(tf.square(u[:, 0:D_EACH]-u[:, D_EACH:D]), axis=1)


# build tensorflow graph
for run_K in range(N_RUNS):
    t0 = time.time()
    tf.reset_default_graph()
    # functions for unrolling
    def remove_original_op_attributes(graph):
        # Remove _original_op attribute from all operations in a graph
        for op in graph.get_operations():
            op._original_op = None
    #
    #
    _graph_replace = tf.contrib.graph_editor.graph_replace
    #
    #
    def graph_replace(*args, **kwargs):
        # Monkey patch graph_replace so that it works with TF 1.0
        remove_original_op_attributes(tf.get_default_graph())
        return _graph_replace(*args, **kwargs)
    #
    #
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
    #
    #
    x_mu = tf.placeholder(shape=[None, D], dtype=tf.float32)
    x_theta = tf.placeholder(shape=[None, D_THETA, N_GEN], dtype=tf.float32)
    inds_gen = tf.placeholder(shape=[None, D, N_GEN],dtype=tf.float32)  # determines which generator is chosen for each
                                                                        # of the samples.
                                                                        # should be constant along the second dimension,
                                                                        # with entries either 0 or 1 so that
                                                                        # inds[i, j, :] sums to one
    y = 0
    for i in range(N_GEN):
        y += inds_gen[:, :, i] * ff_net(x_theta[:, :, i], 'T_'+str(i), input_dim=D_THETA, output_dim=D, activation=ACT_T,
                                    n_layers=LAYERS_T, hidden_dim=HIDDEN_T)
    T_theta = y
    #
    h_mu = 0  # sum over h evaluated at samples of mu
    h_T_theta = 0  # sum over h evaluated at samples of theta
    for i in range(2):
        h_mu += ff_net(x_mu[:, i*D_EACH:(i + 1)*D_EACH], 'h' + str(i), input_dim=D_EACH, output_dim=1, activation=ACT_H,
                       n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
        h_T_theta += ff_net(T_theta[:, i*D_EACH:(i + 1)*D_EACH], 'h' + str(i), input_dim=D_EACH, output_dim=1, activation=ACT_H,
                            n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
    #
    obj = tf.reduce_mean(f_objective(T_theta)) + tf.reduce_mean(h_T_theta) - tf.reduce_mean(h_mu)
    integral = tf.reduce_mean(f_objective(T_theta))
    #
    T_vars = [v for v in tf.compat.v1.global_variables() if ('T' in v.name)]
    h_vars = [v for v in tf.compat.v1.global_variables() if ('h' in v.name)]
    #
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
    #
    train_op_T = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999, epsilon=1e-08).minimize(
        -unrolled_loss, var_list=T_vars)
    #
    # training and saving values
    integral_values = []
    objective_values = []
    samp_mu = sample_mu(BATCH_MU)
    samp_theta = sample_theta(BATCH_THETA)
    samp_ind_gen = inds_generator(BATCH_THETA)
    #
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
        #
        if SAVE_VALUES == 1:
            np.savetxt('output/objective_values_combined_' + str(boc) + '_' + str(HIDDEN) + '_' + str(D_EACH) + '_' + str(run_K), objective_values)
            np.savetxt('output/integral_values_combined_' + str(boc) + '_' + str(HIDDEN) + '_' + str(D_EACH) + '_' + str(run_K), integral_values)
