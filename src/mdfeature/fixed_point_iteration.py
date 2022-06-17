import numpy as np
import matplotlib.pyplot as plt


def update_lambda(old_lambda, counts, stationary_distribution):
    intermediate_matrix = np.zeros(counts.shape)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            intermediate_matrix[i][j] = (counts[i][j] + counts[j][i]) * (stationary_distribution[j] * old_lambda[i]) / (
                        old_lambda[i] * stationary_distribution[j] + old_lambda[j] * stationary_distribution[i])

    return np.sum(intermediate_matrix, axis=0)


def calculate_lagrangian_parameters(counts, stationary_distribution, err, quiet=True):
    lambda_vec = np.random.normal(size=counts.shape[0])
    current_error = float('inf')
    iterations = 0
    while current_error > err:
        new_lambda_vec = update_lambda(lambda_vec, counts, stationary_distribution)
        current_error = np.linalg.norm(new_lambda_vec - lambda_vec)
        lambda_vec = new_lambda_vec
        iterations += 1
        if iterations % 5 == 0 and not quiet:
            print(iterations)

    return lambda_vec


def fit_MSM_from_stationary_distribution(counts, stationary_distribution, err, quiet=True):
    lambda_vec = calculate_lagrangian_parameters(counts, stationary_distribution, err, quiet=quiet)

    transition_matrix = np.zeros(counts.shape)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            transition_matrix[i][j] = (counts[i][j] + counts[j][i]) * (stationary_distribution[i]) / (
                        lambda_vec[i] * stationary_distribution[j] + lambda_vec[j] * stationary_distribution[i])

    return transition_matrix

########################################################################################################################

def dQ2(Q, i):
    return (Q[i + 1] - Q[i]) ** 2


def compute_G(Q, X, x):
    G = np.zeros(X.shape, dtype='float64')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if i > 0:
                G[i][j] += 2 * dQ2(Q, i - 1) ** 2 * X[i - 1][i] ** 2 / (x[i - 1] ** 2 * x[i] ** 3)

            if i > 1:
                G[i][j] -= dQ2(Q, i - 2) * dQ2(Q, i - 1) * X[i - 2][i - 1] / (x[i - 2] * x[i - 1] ** 2 * x[i] ** 2)

            if i < X.shape[0] - 1:
                if i + 1 == j:
                    G[i][j] -= 2 * dQ2(Q, i) ** 2 * X[i][i + 1] / (x[i] ** 2 * x[i + 1] ** 2)
                G[i][j] += 2 * dQ2(Q, i) ** 2 * X[i][i + 1] ** 2 / (x[i] ** 3 * x[i + 1] ** 2)

            if i < X.shape[0] - 2:
                if i + 1 == j:
                    G[i][j] += dQ2(Q, i) * dQ2(Q, i + 1) * X[i + 1][i + 2] / (x[i] * x[i + 1] ** 2 * x[i + 2])
                G[i][j] -= dQ2(Q, i) * dQ2(Q, i + 1) * X[i][i + 1] * X[i + 1][i + 2] / (
                            x[i] ** 2 * x[i + 1] ** 2 * x[i + 2])

            if 0 < i < X.shape[0] - 1:
                if i + 1 == j:
                    G[i][j] += dQ2(Q, i) * dQ2(Q, i - 1) * X[i - 1][i] / (x[i - 1] * x[i] ** 2 * x[i + 1])
                G[i][j] -= 2 * dQ2(Q, i) * dQ2(Q, i - 1) * X[i - 1][i] * X[i][i + 1] / (x[i - 1] * x[i] ** 3 * x[i + 1])

    return G


def compute_F(Q, X, x, gamma):
    G = compute_G(Q, X, x)
    min_deltaQ2 = min([dQ2(Q, i) for i in range(len(Q) - 1)])

    return 1/(min_deltaQ2 * gamma)**2 * (G + G.T)


def update_X(old_X, C, Q, gamma):
    new_X = np.zeros(old_X.shape, dtype='float64')
    old_x = np.sum(old_X, axis=1, dtype='float64')
    F = compute_F(Q, old_X, old_x, gamma)
    c = np.sum(C, axis=1)

    for i in range(old_X.shape[0]):
        for j in range(old_X.shape[1]):
            new_X[i][j] = (C[i][j] + C[j][i]) / (c[i] / old_x[i] + c[j] / old_x[j] + F[i][j])

    return new_X


def compute_error(old_X, new_X):
    return np.linalg.norm(np.sum(new_X, axis=1) - np.sum(old_X, axis=1))


def fit_MSM_with_gamma_smoothing(counts, coordinates, gamma, err):
    old_X = (counts + counts.T) / (2 * np.sum(counts))
    current_err = float('inf')

    iterations = 0
    while current_err > err:
        new_X = update_X(old_X, counts, coordinates, gamma)
        current_err = compute_error(old_X, new_X)
        old_X = new_X
        iterations += 1

    stationary_distribution = np.sum(old_X, axis=1)
    transition_matrix = (old_X.T / stationary_distribution).T

    print(f'Finished in {iterations} iteration(s). Error {round(current_err, 7)}.')

    return stationary_distribution, transition_matrix


if __name__ == "__main__":
    Q = np.array([0, 0.1, 0.2, 0.3])
    C = np.array([[1000, 50, 20, 10], [48, 50, 1, 1], [1, 1, 600, 40], [9, 7, 33, 300]])

    stationary_distribution, transition_matrix = fit_MSM_with_gamma_smoothing(counts=C, coordinates=Q, gamma=1.1,
                                                                              err=0.0001)

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(transition_matrix)
    plt.title(r'$p_{ij}$', fontsize=16)
    plt.xlabel('i', fontsize=16)
    plt.ylabel('j', fontsize=16)
    plt.colorbar()
    plt.show()

    plt.plot(stationary_distribution)
    plt.xlabel('i', fontsize=16)
    plt.ylabel(r'$\pi(i)$', fontsize=16)

    print(stationary_distribution)
    print(np.sum(stationary_distribution))