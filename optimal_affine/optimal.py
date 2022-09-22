import numpy as np
from .basis import affine_basis
from .lie import matrix_to_lie, lie_to_matrix
from .expm import expm


def optimal_affine(matrices, loss='exp', basis='SE'):
    """Compute optimal "template-to-subj" matrices from "subj-to-subj" pairs.

    Parameters
    ----------
    matrices : dict(tuple[int, int] -> (D+1, D+1) array)
        Affine matrix for each pair
    loss : {'exp', 'log'}
        Whether to minimize the L2 loss in the embedding matrix space
        ('exp') or in the Lie algebra ('log').
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
    optimal_parameters = log_optimal(parameters)
    if loss[0].lower() == 'm':
        optimal_parameters = exp_optimal(matrices, basis=basis,
                                         init=optimal_parameters)
    optimal_matrices = lie_to_matrix(optimal_parameters, basis)
    return optimal_matrices


def log_optimal(parameters):
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


def exp_optimal(matrices, basis, init=None):
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

    def dot(x, y):
        return np.matmul(x[..., None, :], y[..., None])[..., 0, 0]

    labels = list(set([label for pair in matrices for label in pair]))
    n = len(labels)
    f = len(basis)
    if init is None:
        init = np.zeros(n, len(basis))
    z = np.copy(init)

    x = flat(np.stack(list(matrices.values()), -3))
    keys = [(labels.index(i), labels.index(j)) for i, j in matrices.keys()]

    loss_prev = float('inf')
    for n_iter in range(1000):
        y, gz, hz = expm(z, basis, grad_X=True, hess_X=True)
        iy, igz, ihz = expm(-z, basis, grad_X=True, hess_X=True)

        h = np.zeros([n, f, n, f])
        g = np.zeros([n, f])
        loss = 0

        for (i, j), xij in zip(keys, x):
            yij = flat(np.matmul(y[i], iy[j]))      # [D*D]
            r = yij - xij                           # [D*D]
            loss += dot(r, r) / len(r)

            gi = flat(np.matmul(gz[i], iy[j]))      # [F, D*D]
            hi = np.matmul(hz[i], iy[j])            # [F, F, D, D]
            hi = flat(np.abs(hi).sum(-3))           # [F, D*D]
            hii = np.matmul(gi, gi.T)               # [F, F]

            gj = -flat(np.matmul(y[i], igz[j]))     # [F, D*D]
            hj = np.matmul(y[i], ihz[j])            # [F, F, D, D]
            hj = flat(np.abs(hj).sum(-3))           # [F, D*D]
            hjj = np.matmul(gj, gj.T)               # [F, F]

            hij = np.matmul(gi, gj.T)               # [F, F]

            g[i, :] += dot(gi, r)                           # [F]
            hii[range(f), range(f)] += dot(hi, np.abs(r))   # [F]
            hii[range(f), range(f)] *= 1.001
            h[i, :, i, :] += hii

            g[j, :] += dot(gj, r)                           # [F]
            hjj[range(f), range(f)] += dot(hj, np.abs(r))   # [F]
            hjj[range(f), range(f)] *= 1.001
            h[j, :, j, :] += hjj

            h[i, :, j, :] += hij
            h[j, :, i, :] += hij

        loss /= len(x)
        print(loss, loss_prev - loss)

        g = g.reshape([n*f])
        h = h.reshape([n*f, n*f])
        delta = np.linalg.solve(h, g[..., None])[..., 0]    # [N*F]
        delta = delta.reshape([n, f])
        z -= delta
        z -= np.mean(z, axis=-2, keepdims=True)             # zero-center

        if loss_prev - loss < 1e-9:
            break
        loss_prev = loss

    return z
