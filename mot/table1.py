import numpy as np


cases = ['base', 'mixture', 'unrolling', 'combined']
N_RUNS = 10     # to save space, the uploaded values only contain output of the median (in terms of integral value) runs
                # hence the values differ to the ones given in Table 1 in the paper

def f_np(x):
    return np.maximum(x[:, 1] - x[:, 0], 0)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


RANGE_MIN = -6
RANGE_MAX = 6


def test_fun(i, x):
    c_h = np.zeros(i+1)
    c_h[i] = 1
    return np.polynomial.chebyshev.chebval((x-RANGE_MIN)*(2/(RANGE_MAX-RANGE_MIN))-1, c_h)


mean1 = -1.3
mean2 = 0.8
sig1 = 0.5
sig2 = 0.7
sig3 = 1.1
sig4 = 1.3
mixt = 0.5


def sample_mu(K):
    points1 = np.random.randn(K)
    id = np.random.binomial(1, mixt, size=K)
    points1 = points1 * (id * sig1 + (1-id) * sig2) + id * mean1 + (1-id) * mean2
    er2 = np.random.randn(K)
    points2 = points1 + er2 * ((sig3 - sig1) * id + (sig4 - sig2) * (1-id))
    out = np.zeros([K, 2])
    out[:, 0] = points1
    out[:, 1] = points2
    return out


real_samples = sample_mu(100000)


for case in cases:
    average_integral_value = 0
    average_marginal_error = 0
    average_martingale_error = 0
    average_stdev = 0
    av_counter = 0
    for run_K in range(N_RUNS):
        # load saved samples and values if possible
        try:
            values = np.loadtxt('output/objective_values_' + case + str(run_K))
            samples = np.loadtxt('output/samples_' + case + str(run_K))
            av_counter += 1
        except:
            print('For case ' + case + ' and run ' + str(run_K) + ' no saved data is available')
            continue
        integral_value = np.mean(f_np(samples))
        average_integral_value += integral_value

        marginals_error = 0
        martingale_error = 0
        for i in range(50):
            for j in range(2):
                marginals_error += (1/2) * (1/50) * np.abs(np.mean(test_fun(i, samples[:, j])) -
                                                           np.mean(test_fun(i, real_samples[:, j])))
            martingale_error += (1/50) * np.abs(np.mean(test_fun(i, samples[:, 0]) * (samples[:, 1] - samples[:, 0])))

        average_marginal_error += marginals_error
        average_martingale_error += martingale_error

        stdev = np.std(values[-2500:])
        average_stdev += stdev
    if av_counter > 0:
        average_integral_value /= av_counter
        average_marginal_error /= av_counter
        average_martingale_error /= av_counter
        average_stdev /= av_counter
        print('Values for Table 1 for case ' + str(case))
        print('Integral value = ' + str(average_integral_value))
        print('Marginal error = ' + str(average_marginal_error))
        print('Martingale error = ' + str(average_martingale_error))
        print('Std dev iterations = ' + str(average_stdev))


