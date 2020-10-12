import os, sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

N_inf_values = [1, 10]
cases = ['base', 'divergence', 'lipschitz']
N_RUNS = 10
N_r = 500


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


for case in cases:
    print(case)
    for N_inf in N_inf_values:
        print(N_inf)
        std_devs = []
        for run_K in range(N_RUNS):
            # load saved samples and values if possible
            try:
                values = np.loadtxt('output/objective_values_' + case + str(N_inf) + '_' + str(run_K))
            except:
                print('For case ' + case + ' with N_inf = ' + str(N_inf) + ' and run '
                      + str(run_K) + ' no saved data is available')
                continue
            rm = running_mean(values, N_r)
            std_dev = np.std(rm[-5000:])
            std_devs.append(std_dev)

        if len(std_devs) > 0:
            medIdx = std_devs.index(np.percentile(std_devs, 50, interpolation='nearest'))
            print('Median run index: ' + str(medIdx))
            median_values = np.loadtxt('output/objective_values_' + case + str(N_inf) + '_' + str(medIdx))
            median_values_rm = running_mean(median_values, N_r)
            if case == 'base' and N_inf == 1:
                rm_base0 = median_values_rm
            if case == 'base' and N_inf == 10:
                rm_base1 = median_values_rm
            if case == 'divergence' and N_inf == 1:
                rm_pen0 = median_values_rm
            if case == 'divergence' and N_inf == 10:
                rm_pen1 = median_values_rm
            if case == 'lipschitz' and N_inf == 1:
                rm_spec0 = median_values_rm
            if case == 'lipschitz' and N_inf == 10:
                rm_spec1 = median_values_rm

# Plot
plt.rc('text', usetex=True)

plt.figure(figsize=(16*3.4/5, 9*3.4/5))

ax1 = plt.subplot(2, 3, 1)
ax1.set_xticklabels([])
plt.plot(rm_base0)
plt.ylim((-2, 4))
plt.grid(color='black', linestyle='-.', linewidth=0.2)
plt.title(r'$(P^{m}), N_{inf} = 1$', fontsize=16)
plt.ylabel(r'Objective value', fontsize=16)

ax2 = plt.subplot(2, 3, 2)
ax2.set_xticklabels([])
plt.plot(rm_pen0)
plt.grid(color='black', linestyle='-.', linewidth=0.2)
plt.ylim((-2, 4))
plt.title(r'$(P_{\psi}^{m}), N_{inf} = 1$', fontsize=16)

ax3 = plt.subplot(2, 3, 3)
plt.ylim((-2, 4))
ax3.set_xticklabels([])
plt.grid(color='black', linestyle='-.', linewidth=0.2)
plt.title(r'$(P_{L}^{m}), N_{inf} = 1$', fontsize=16)
plt.plot(rm_spec0)

ax4 = plt.subplot(2, 3, 4)
plt.ylim((-2, 4))
plt.plot(rm_base1, label='Base')
plt.ylabel(r'Objective value', fontsize=16)
plt.xlabel(r'$N_{iter}$', fontsize=16)
plt.grid(color='black', linestyle='-.', linewidth=0.2)
plt.title(r'$(P^{m}), N_{inf} = 10$', fontsize=16)

ax5 = plt.subplot(2, 3, 5)
plt.ylim((-2, 4))
plt.plot(rm_pen1)
plt.xlabel(r'$N_{iter}$', fontsize=16)
plt.grid(color='black', linestyle='-.', linewidth=0.2)
plt.title(r'$(P_{\psi}^{m}), N_{inf} = 10$', fontsize=16)

ax6 = plt.subplot(2, 3, 6)
plt.ylim((-2, 4))
plt.title(r'$(P_{L}^{m}), N_{inf} = 10$', fontsize=16)
plt.plot(rm_spec1)
plt.xlabel(r'$N_{iter}$', fontsize=16)
plt.grid(color='black', linestyle='-.', linewidth=0.2)
plt.savefig('output/figure1.png', format='png', dpi=300)
plt.show()
