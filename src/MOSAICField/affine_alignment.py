import numpy as np
import ot
import torch
from scipy.linalg import lstsq
from scipy.spatial import distance


def lstsq_proj_mat(sp1, sp2, pi):
    """
    Uses Least Squares to calculate an affine transformation.
    The 6 free parameters denote the upper two rows of R.
    The expected transformation is Y = RX.
    """
    nnz = np.sum(pi > 0)
    A = np.zeros((2 * nnz, 6))
    B = np.zeros(2 * nnz)
    idx = 0
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            if pi[i][j] > 0:
                w = pi[i][j] ** 0.5
                x1, y1 = sp1[i]
                x2, y2 = sp2[j]
                A[idx] = [x1 * w, y1 * w, w, 0, 0, 0]
                A[idx + 1] = [0, 0, 0, x1 * w, y1 * w, w]
                B[idx] = x2 * w
                B[idx + 1] = y2 * w
                idx += 2
    return A, B


def lstsq_proj(sp1, sp2, pi):
    A, B = lstsq_proj_mat(sp1, sp2, pi)
    T, _, _, _ = lstsq(A, B)
    return np.append(T, [0, 0, 1]).reshape(3, 3)


def affine_transformation(X, Y, P):
    T = lstsq_proj(sp1=X, sp2=Y, pi=P)
    homogeneous_X = np.vstack([X.T, np.ones((1, X.shape[0]))])
    transformed_homogeneous_X = T @ homogeneous_X
    return transformed_homogeneous_X[:2, :].T, T


def FGW_affine(X, Y, FX, FY, max_iter=100, alpha=0.8, device="cpu"):
    # center
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    D_X = distance.cdist(FX, FX)
    D_Y = distance.cdist(FY, FY)
    D_X /= D_X.max()
    D_Y /= D_Y.max()
    D_X = torch.tensor(D_X).to(device)
    D_Y = torch.tensor(D_Y).to(device)

    T_net = np.eye(3)
    for iter in range(max_iter):
        print("Iter:", iter)
        C = distance.cdist(X, Y)
        C /= C[C > 0].max()
        C = torch.tensor(C).to(device)

        # FGW
        P = ot.gromov.fused_gromov_wasserstein(C, D_X, D_Y, alpha=alpha).to(
            torch.float32
        )
        # Affine Transformation
        X, T = affine_transformation(X=X, Y=Y, P=P.cpu().numpy())

        T_net = T @ T_net

    # Compute final P
    C = distance.cdist(X, Y)
    C /= C[C > 0].max()
    C = torch.tensor(C).to(device)
    P = ot.gromov.fused_gromov_wasserstein(C, D_X, D_Y, alpha=alpha).to(torch.float32)
    return X, Y, T_net, P


def minibatch_affine_transformation(X_batch, Y_batch, P, X):
    T = lstsq_proj(sp1=X_batch, sp2=Y_batch, pi=P)
    homogeneous_X = np.vstack([X.T, np.ones((1, X.shape[0]))])
    transformed_homogeneous_X = T @ homogeneous_X
    return transformed_homogeneous_X[:2, :].T, T


def minibatch_FGW_affine(
    X, Y, FX, FY, batch_size=3000, max_iter=10, alpha=0.8, device="cpu"
):
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    n_dim = X.shape[1]

    # Center X onto Y (centroid alignment)
    mu_X = X.mean(axis=0, keepdims=True)
    mu_Y = Y.mean(axis=0, keepdims=True)
    delta = (mu_Y - mu_X).ravel()
    X = X + delta

    T_center = np.eye(n_dim + 1, dtype=np.float32)
    T_center[:-1, -1] = delta
    T_net = T_center.copy()

    # Iterative FGW + affine refinement
    for it in range(max_iter):
        print("Iter:", it)

        X_batch_indices = np.random.choice(X.shape[0], batch_size, replace=False)
        Y_batch_indices = np.random.choice(Y.shape[0], batch_size, replace=False)
        X_batch = X[X_batch_indices]
        Y_batch = Y[Y_batch_indices]

        C = distance.cdist(X_batch, Y_batch)
        C /= C[C > 0].max()
        C = torch.tensor(C, dtype=torch.float32, device=device)

        FX_batch = FX[X_batch_indices]
        FY_batch = FY[Y_batch_indices]
        D_X = distance.cdist(FX_batch, FX_batch)
        D_Y = distance.cdist(FY_batch, FY_batch)
        D_X /= D_X.max()
        D_Y /= D_Y.max()
        D_X = torch.tensor(D_X, dtype=torch.float32, device=device)
        D_Y = torch.tensor(D_Y, dtype=torch.float32, device=device)

        P = ot.gromov.fused_gromov_wasserstein(
            C, D_X, D_Y, p=None, q=None, alpha=alpha
        ).to(torch.float32)

        X, T = minibatch_affine_transformation(
            X_batch=X_batch, Y_batch=Y_batch, P=P.cpu().numpy(), X=X
        )

        T_net = T @ T_net

    return X, Y, T_net


def affine_align(slice1, slice2, obsm_name=None, max_iter=10, alpha=0.8):
    """
    Optimal affine transformation between two slices.

    param: slice1 - AnnData object of slice 1
    param: slice2 - AnnData object of slice 2
    param: obsm_name - obsm field for features. If None, use slice.X
    param: max_iter - Max iterations for optimal transport
    param: alpha - Balance parameter. 0 <= alpha <= 1

    return: slice1, slice2 - Affine aligned AnnData objects.
        slice1.obsm['spatial'] is updated.
    return: T - 3x3 affine transformation matrix.
    return: P - Probabilistic mapping between locations.
    """
    X = slice1.obsm["spatial"].astype(np.float32)
    Y = slice2.obsm["spatial"].astype(np.float32)
    if obsm_name is None:
        FX = slice1.X.astype(np.float32)
        FY = slice2.X.astype(np.float32)
    else:
        FX = slice1.obsm[obsm_name].astype(np.float32)
        FY = slice2.obsm[obsm_name].astype(np.float32)

    X, Y, T, P = FGW_affine(X, Y, FX, FY, max_iter=max_iter, alpha=alpha)

    slice1_copy = slice1.copy()
    slice2_copy = slice2.copy()
    slice1_copy.obsm["spatial"] = X
    return slice1_copy, slice2_copy, T, P
