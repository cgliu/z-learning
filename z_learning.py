import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import animation
from random import choices


def viz_mat(M, ax=None, alpha=1.0, annot=False):
    """
    Visualize 2d tensor M.
    """
    if ax is None:
        _, ax = plt.subplots()
    return sns.heatmap(M, ax=ax, cbar=False, square=True, alpha=alpha, annot=annot)


def viz_state_prob(M, state_prob_history, blit=False, save=False):
    """
    Visualize the history of state probability.

    Args:
    M (tensor): the map tensor.
    state_prob_history (list of tensors): the history of state probability.
    blit (bool): True to optimize drawing using z buffer.
    save (bool): True to save animation as anim.mp4
    """
    num_rows, num_cols = M.size()
    num_steps = len(state_prob_history)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout()
    viz_mat(M, ax, alpha=0.5)

    def init():
        patch = plt.Circle((0.5, 0.5),
                           0.5,
                           fc='red',
                           ec=None,
                           lw=1, alpha=0.5)
        ax.add_patch(patch)
        return patch,

    def animate(i):
        state_prob = state_prob_history[i].squeeze()
        state_inds = torch.arange(state_prob.size()[0])[state_prob > 1e-2]
        patches = []
        for ind in state_inds.tolist():
            row = int(ind / num_cols)
            col = ind - row * num_cols
            patch = plt.Circle((col+0.5, row+0.5),
                               0.5,
                               fc='red',
                               ec=None,
                               lw=1,
                               alpha=state_prob[ind].item())
            ax.add_patch(patch)
            patches.append(patch)
        return patches

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=num_steps,
                                   interval=50,
                                   blit=blit,
                                   repeat=True)

    if save:
        writervideo = animation.FFMpegWriter(fps=60)
        anim.save("anim.mp4", writer=writervideo)
    plt.show()


def simulation(M, callback, num_steps=100, blit=False, save=False, repeat=False, annot=False):
    """
    Interactive simulation.
    Args:
    M (2d tensor): the Map.
    callback: callback function, which should return state probability and
              its desirability.
    num_step (int): number of simulation steps
    blit (bool): True to optimize drawing using z buffer.
                 Refer to FuncAnimation().
    save (bool): True to save anim.mp4.
    repeat (bool): True to repeat simulation.
                   Refer to FuncAnimation().
    annot (bool): True to show annotation.
    """

    num_rows, num_cols = M.size()
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    fig.tight_layout()
    viz_mat(M, axs[0], alpha=0.5)
    axs[0].add_patch(plt.Rectangle((num_cols - 1, num_rows - 1),
                                   1, 1,
                                   fc='y',
                                   ec=None,
                                   lw=1,
                                   alpha=0.5))

    def init():
        patches = []
        patches.append(
            plt.Circle((0.5, 0.5),
                       0.5,
                       fc='red',
                       ec=None,
                       lw=1,
                       alpha=0.5))
        for p in patches:
            axs[0].add_patch(p)
        return patches

    def animate(frame_ind):
        axs[1].clear()
        axs[0].clear()
        viz_mat(M, axs[0], alpha=0.5)

        state_prob, z = callback(frame_ind)
        patches = []
        viz_mat(-torch.log(z).reshape(M.size()), axs[1], alpha=1.0, annot = annot)

        state_inds = torch.arange(state_prob.size()[0])[state_prob > 1e-2]
        for ind in state_inds.tolist():
            row = int(ind / num_cols)
            col = ind - row * num_cols
            fc = 'red' if ind < z.size()[0] - 1 else "green"
            patch = plt.Circle((col+0.5, row+0.5),
                               0.5,
                               fc=fc,
                               ec=None,
                               lw=1,
                               alpha=state_prob[ind].item())
            axs[0].add_patch(patch)
            patches.append(patch)
        # fig.canvas.draw()
        # renderer = fig.canvas.renderer
        # axs[1].draw(renderer)

        return patches

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=num_steps,
                                   interval=50,
                                   blit=blit,
                                   repeat=repeat)

    if save:
        writervideo = animation.FFMpegWriter(fps=30)
        anim.save("anim.mp4", writer=writervideo)
    plt.show()


def get_neighbor_indices(query, rows, cols):
    """
    Get the indices of four neighbor.
    Args:
    query (int): contiguous query index.
    rows (int): number of rows.
    cols (int): number of columns.
    Returns:
    a list of contiguous indices of your neighbors.
    """
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
    """
    Create a 2D map with random occupancy. The top right corner and the bottom
    row are always clear to increase the connectivity to the goal.

    Args:
    occupancy (float): percentage of occupancy.

    Returns:
    a 2D tensor.
    """
    torch.manual_seed(0)
    M = torch.rand(rows, cols)
    M[M <= occupancy] = 0
    M[M > 0] = 1
    # start from top left
    M[0, 0] = 1
    # end at bottom righ
    M[-1][-1] = 1
    M[-1][:] = 1
    return M


def create_transition_matrix(M, be_trapped=False):
    """
    Given a map matrix M, create transition matrix P
    If M is M by N, P will be M * N by M * N
    p_{ij} is the transition probability from i to j of M.flatten
    """
    rows, cols = M.size()
    P = torch.zeros(rows * cols, rows * cols)
    for i in range(P.size()[0]):
        neighbor_indices = get_neighbor_indices(i, rows, cols)
        for j in neighbor_indices:
            P[i, j] = M[int(j / cols), j - int(j / cols) * cols]

    # Make sure that all rows are non-zero. If there is no feasible neighbor,
    # 'setting pii to be one' means the robot will stay at state i for ever.
    for i in range(rows * cols):
        if P[i, :].max() == 0:
            P[i, i] = 1.0
    # Create absorting state at the bottom right corner
    if be_trapped:
        P[-1, :] = 0
        P[-1, -1] = 1
    return P / P.sum(dim=1, keepdim=True)


def sto_solver(P, zf, q, max_num_iters=10):
    """
    Solve the MPDs.
    Args:
    P (tensor): the passive state transition matrix.
    zf (tensor): the initial state desirability.
    max_num_iters (int): the maximum number of iterations.

    Returns:
    z (tensor): the optimal state desirability.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    P_ = P.clone().detach().to(device)
    p = torch.exp(-q).to(device)
    G = torch.diag(p.squeeze()).to(device)

    assert G.size() == P.size(
    ), f"G.size() = {G.size()} vs P.size() = {P.size()}"
    z = torch.t(zf).to(device)

    iter = 0
    converged = False
    while iter < max_num_iters:
        z_new = G @ P_ @ z
        diff = (z - z_new).abs().max()
        converged = (diff < 1e-20)
        if iter % 100 == 0:
            print(f"{iter}: {diff}")
        z = z_new
        if converged:
            break

        iter += 1
    return torch.t(z).to(P.device)


def z_learn(z0, replay_buffer, alpha):
    """
    In Z-learning, we assume the dynamics (P) is unknown.
    All we have access to are samples(i, j, q).
    """
    z = zo.clone().detach().squeeze()
    for i, j, q in replay_buffer:
        z[i] = (1 - alpha) * z(i) + alpha * torch.exp(-q) * z[j]

    return z.reshape(z0.size())


def single_rollout(x0, P, rho, num_steps=10):
    """
    Create a single rollout.
    
    Args:
    x0 (tensor): the initial state distribution.
    P (tensor): the state transition matrix.
    num_steps (int): the number of rollout steps.

    Returns:
    rollout (tensor): a list of state tensor.
    """
    x = x0.clone().detach()
    indices = torch.arange(P.size()[0])
    q = rho
    rollout = []
    for step in range(num_steps):
        x_next = x @ P
        # During rollout, we can only observe one sample from the underlining
        # distribution. collapse prob into one hot here
        j = choices(indices, x_next[0, :])
        x_next = torch.zeros_like(x0)
        x_next[0, j] = 1.0
        rollout.append([x.clone().detach(), x_next.clone().detach(), q])
        x = x_next

    return rollout
