import numpy as np
from .basis import affine_basis
from .lie import matrix_to_lie, lie_to_matrix
from .expm import expm


def optimal_affine(matrices, loss='matrix', basis='SE'):
    """Compute optimal "template-to-subj" matrices from "subj-to-subj" pairs.

    Parameters
    ----------
    matrices : dict(tuple[int, int] -> (D+1, D+1) array)
        Affine matrix for each pair
    loss : {'matrix', 'lie'}
        Whether to minimize the L2 loss in the embedding matrix space
        or in the Lie algebra.
    basis : {'SE', 'CSO', 'Aff+'}, default='SE'
        Constrain matrices to belong to this group

    Returns
    -------
    optimal : (N, D+1, D+1) array
        Optimal template-to-subject matrices

    """
    ndim = list(matrices.values())[0].shape[-1] - 1
    basis = affine_basis(basis, ndim, dtype=list(matrices.values())[0].dtype)
    parameters = {k: matrix_to_lie(v, basis)[0] for k, v in matrices.items()}
    optimal_parameters = lie_optimal(parameters)
    if loss[0].lower() == 'm':
        optimal_parameters = matrix_optimal(matrices, basis=basis,
                                            init=optimal_parameters)
    optimal_matrices = lie_to_matrix(optimal_parameters, basis)
    return optimal_matrices


def lie_optimal(parameters):
    """Compute optimal Lie parameters wrt L2 loss in the Lie algebra

    Parameters
    ----------
    parameters : dict(tuple[int, int] -> (F,) array)
        Lie parameters for each pair

    Returns
    -------
    optimal : (N, F) array
        Optimal template-to-subject parameters

    """
    labels = list(set([label for pair in parameters for label in pair]))
    n = len(labels)
    w = np.zeros([len(parameters), n])
    for p, (i, j) in enumerate(parameters.keys()):
        w[p, labels.index(i)], w[p, labels.index(j)] = 1, -1
    x = np.stack(list(parameters.values()), -2)
    w = np.linalg.pinv(w)
    z = np.matmul(w, x)
    z -= np.mean(z, axis=-2, keepdims=True)  # zero-center
    return z


def matrix_optimal(matrices, basis, init=None):
    """Compute optimal Lie parameters wrt L2 loss in the embedding space

    Parameters
    ----------
    matrices : dict(tuple[int, int] -> (D+1, D+1) array)
        Affine matrix for each pair
    basis : (F, D+1, D+1)
        Constrain matrices to belong to this group
    init : (N, F) array
        Initial guess

    Returns
    -------
    optimal : (N, F) array
        Optimal template-to-subject parameters

    """
    def flat(x):
        return x.reshape(x.shape[:-2] + (-1,))

    def flat2(x):
        x = flat(x)
        return x.reshape(x.shape[:-3] + (-1, x.shape[-1]))

    def t(x):
        return np.swapaxes(x, -1, -2)

    labels = list(set([label for pair in matrices for label in pair]))
    n = len(labels)
    f = len(basis)
    if init is None:
        init = np.zeros(n, len(basis))
    z = np.copy(init)

    w = np.zeros([len(matrices), n])
    for p, (i, j) in enumerate(matrices.keys()):
        w[p, labels.index(i)], w[p, labels.index(j)] = 1, -1
    ww = np.matmul(w.T, w)
    x = flat(np.stack(list(matrices.values()), -3))

    loss_prev = float('inf')
    for n_iter in range(1000):
        y, gz, hz = expm(z, basis, grad_X=True, hess_X=True)
        gz = flat(gz)                                        # [N, F, D*D]
        hz = flat(np.abs(hz).sum(-3))                        # [N, F, D*D]
        ggz = np.matmul(gz[..., None, None, None, :],        # [N, F, 1, 1, 1, D*D]
                        gz[..., None, None, :, :, :, None])  # [1, 1, N, F, D*D, 1]
        ggz = ggz[..., 0, 0]                                 # [N, F, N, F]

        r = np.matmul(w, flat(y)) - x                   # [P, D*D]
        loss = np.dot(r.flatten(), r.flatten())
        if loss_prev - loss < 1e-9:
            break
        loss_prev = loss

        r = np.matmul(w.T, r)                           # [N, D*D]
        g = np.matmul(gz, r[..., None])[..., 0]         # [N, F]
        h = ggz * ww[:, None, :, None]                  # [N, F, N, F]
        h = flat2(h)                                    # [N*F, N*F]

        hz = np.matmul(hz[..., None, :],                # [N, F, 1, D*D]
                       np.abs(r[..., None, :, None]))   # [N, 1, D*D, 1]
        hz = hz[..., 0, 0]                              # [N, F]
        h[..., np.arange(n*f), np.arange(n*f)] += flat(hz)
        h[..., np.arange(n * f), np.arange(n * f)] *= 1.001

        g = flat(g)
        delta = np.linalg.solve(h, g[..., None])[..., 0]  # [N*F]
        delta = delta.reshape([n, f])
        z -= delta
        z -= np.mean(z, axis=-2, keepdims=True)  # zero-center

    return z
