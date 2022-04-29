# import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def viz_mat(M, ax=None, alpha=1.0, annot=False):
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(M, ax=ax, cbar=False, square=True, alpha=alpha, annot=annot)


def viz_P(P):
    sns.heatmap(P, annot=True)


def get_neighbor_indices(query, rows, cols):
    # 4 neighbors
    row = int(query / cols)
    col = query - row * cols

    neighbor_indices = []
    for r, c in [[row, col - 1],
                 [row, col + 1],
                 [row - 1, col],
                 [row + 1, col]]:
        if 0 <= r and r < rows and 0 <= c and c < cols:
            neighbor_indices.append(r * cols + c)
    neighbor_indices.sort()
    return neighbor_indices


def create_grid_map(rows, cols, occupancy=0.3):
    torch.manual_seed(0)
    M = torch.rand(rows, cols)
    M[M <= occupancy] = 0
    M[M > 0] = 1
    # start from top left
    M[0, 0] = 1
    # end at bottom righ
    M[-1][-1] = 1
    # increase connectivity to the goal
    M[-1][:] = 1
    return M


def create_transition_matrix(M, be_trapped=False):
    """
    Given a map matrix M, create transition matrix P
    If M is M by N, P will be M * N by M * N
    p_{ij} is the transition probability from i to j of M.flatten
    """
    rows, cols = M.size()
    P = torch.eye(rows * cols, rows * cols)
    for i in range(P.size()[0]):
        neighbor_indices = get_neighbor_indices(i, rows, cols)
        for j in neighbor_indices:
            P[i, j] = M[int(j / cols), j - int(j / cols) * cols]

    # Trap at the bottom right corner
    if be_trapped:
        P[-1, :] = 0
        P[-1, -1] = 1
    return P / P.sum(dim=1, keepdim=True)


def z_learning(P, zf, q, max_num_iters=10):
    """ 
    """
    p = torch.exp(-q)
    G = torch.diag(p.squeeze())

    assert G.size() == P.size(
    ), f"G.size() = {G.size()} vs P.size() = {P.size()}"
    # right product!
    z = torch.t(zf)
    iter = 0
    converged = False
    while iter < max_num_iters:
        z_new = G @ P @ z
        diff = (z - z_new).norm()
        converged = diff < 1e-20
        z = z_new
        if converged:
            break
        if iter % 100 == 0:
            print(f"{iter}: {diff}")

        iter += 1
    return z