import numpy as np
import pyemma
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from mdfeature.MarkovStateModel import MSM
import mdfeature.general_utils as gutl


def implied_timescale_analysis(discrete_traj, lags):
    print('MSM Implied Timescale Analysis')
    its = pyemma.msm.its(discrete_traj, lags=lags, nits=10, reversible=True, connected=True)
    pyemma.plots.plot_implied_timescales(its, ylog=False)
    plt.show()


def diffusion_coefficient_sensitivity_analysis(cluster_centers, discrete_traj, lags, time_step):
    msm = MSM(cluster_centers)
    old_D = None
    diffs = []
    diffusion_coefficients = []
    for lag in lags:
        pyemma_msm = pyemma.msm.estimate_markov_model(dtrajs=discrete_traj, lag=lag)
        msm.set_stationary_distribution(pyemma_msm.stationary_distribution)
        msm.set_transition_matrix(pyemma_msm.transition_matrix)
        D = msm.compute_diffusion_coefficient(time_step, lag)
        diffusion_coefficients.append(D)
        if old_D is not None:
            diffs.append(gutl.vector_rmsd(old_D, D))
        old_D = D

    index_optimal_lag = np.argmin(diffs)
    optimal_lag = lags[index_optimal_lag]

    plt.plot(lags[:-1], diffs, c='k')
    plt.xlabel('Lag', fontsize=16)
    plt.ylabel(r'RMSD $D(Q)$', fontsize=16)
    plt.vlines(optimal_lag, min(diffs), max(diffs), color='r')
    plt.show()
    for idx, diff_coeffs in enumerate(diffusion_coefficients):
        plt.plot(msm.sorted_state_centers, diff_coeffs, label="lag=" + str(lags[idx]))
    plt.legend()
    plt.xlabel(r'$Q$', fontsize=16)
    plt.ylabel(r'$D(Q)$', fontsize=16)
    plt.show()
    # Perform Langevin dynamics check
    for idx, diff_coeffs in enumerate(diffusion_coefficients):
        c4 = msm.calculate_correlation_coefficient(n=4)
        tau = lags[idx] * time_step
        D4 = (1 / (4 * 3 * 2 * tau)) * c4
        error_ratio = D4 / diff_coeffs ** 2
        if min(error_ratio) < 0.25:
            plt.plot(msm.sorted_state_centers, D4 / diff_coeffs ** 2, label="lag=" + str(lags[idx]))
    plt.yscale('log')
    plt.hlines(0.25, min(msm.sorted_state_centers), max(msm.sorted_state_centers), colors='r')
    plt.title("Langevin Dynamics Check", fontsize=16)
    plt.ylabel(r"$D^{(4)}(Q)/D^{(2)}(Q)^2$", fontsize=16)
    plt.xlabel(r"$Q$", fontsize=16)
    plt.legend()
    plt.show()


def lag_sensitivity_analysis(discrete_traj, cluster_centers, time_step):
    lags = [1, 2, 3, 5, 7, 9, 12, 15, 19, 24, 30, 37, 45]
    implied_timescale_analysis(discrete_traj, lags)
    diffusion_coefficient_sensitivity_analysis(cluster_centers, discrete_traj, lags, time_step)


def cluster_time_series(time_series, cluster_type, options):
    if cluster_type == 'kmeans':
        cluster = pyemma.coordinates.cluster_kmeans(time_series,
                                                    k=options['k'],
                                                    stride=options['stride'],
                                                    max_iter=options['max_iter'])
    elif cluster_type == 'reg_space':
        cluster = pyemma.coordinates.cluster_regspace(time_series, dmin=options['dmin'],
                                                      max_centers=options['max_centers'])
    else:
        raise ValueError('cluster_type must be either "kmeans" or "reg_space"')

    return cluster


def compute_well_integrand(free_energy, beta):
    return [np.exp(- beta * free_energy[x]) for x in range(len(free_energy))]


def compute_barrier_integrand(evaluator_object, free_energy, beta):
    return [np.exp(beta * free_energy[x])
            / gutl.linear_interp_coordinate_data(evaluator_object.smoothed_D_domain,
                                                 evaluator_object.smoothed_D,
                                                 evaluator_object.coordinates[x]) for x in range(len(free_energy))]


def compute_well_and_barrier_integrals(initial_x, final_x, mid_x, well_integrand, barrier_integrand, x_coords):
    if final_x > initial_x:
        well_integral = integrate.simpson(well_integrand[initial_x: mid_x + 1], x_coords[initial_x: mid_x + 1])
        barrier_integral = integrate.simpson(barrier_integrand[initial_x + 1: final_x],
                                             x_coords[initial_x + 1:final_x])
    else:
        well_integral = integrate.simpson(well_integrand[mid_x: initial_x + 1], x_coords[mid_x: initial_x + 1])
        barrier_integral = integrate.simpson(barrier_integrand[final_x + 1: initial_x],
                                             x_coords[final_x + 1: initial_x])

    return well_integral, barrier_integral


def compute_kramers_rate(transition, minima, well_integrand, barrier_integrand, x_coords):
    initial_x = minima[transition[0]]
    final_x = minima[transition[1]]
    mid_x = int(np.floor((initial_x + final_x) / 2))
    well_integral, barrier_integral = compute_well_and_barrier_integrals(initial_x,
                                                                         final_x,
                                                                         mid_x,
                                                                         well_integrand,
                                                                         barrier_integrand,
                                                                         x_coords)
    kramers_rate = (barrier_integral * well_integral) ** (-1)

    return kramers_rate

# Chapman-Kolmogorov Test
# if self.verbose:
# print('MSM Chapman-Kolmogorov Test')
# ck_test = msm.cktest(min(msm.nstates, 4))
# pyemma.plots.plot_cktest(ck_test)
# plt.show()

# def _compute_counts_matrix(self, traj, lag):
#     num_of_states = self.msm.number_of_states
#     counts = np.zeros((num_of_states, num_of_states))
#     for idx, state in enumerate(traj[:-lag]):
#         counts[state, traj[idx+lag]] += 1
#
#     return counts
