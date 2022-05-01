import unittest

import z_learning as zl

import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from random import choices


def viz_state_prob(M, state_prob_history, blit=False, save=False):
    num_rows, num_cols = M.size()
    num_steps = len(state_prob_history)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout()
    zl.viz_mat(M, ax, alpha=0.5)

    def init():
        patch = plt.Rectangle((0, 0),
                              1, 1,
                              fc='red',
                              ec=None,
                              lw=1)
        ax.add_patch(patch)
        return patch,

    def animate(i):
        state_prob = state_prob_history[i].squeeze()
        state_inds = torch.arange(state_prob.size()[0])[state_prob > 1e-2]
        patches = []
        for ind in state_inds.tolist():
            row = int(ind / num_cols)
            col = ind - row * num_cols
            patch = plt.Rectangle((col, row),
                                  1, 1,
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

    # anim.save("anim.gif", writer="imagemagick")
    if save:
        writervideo = animation.FFMpegWriter(fps=60)
        anim.save("anim.mp4", writer=writervideo)
    plt.show()


class TestZLearning(unittest.TestCase):
    def my_setup(self, rows=5, cols=5, occupancy=0.3):
        M = zl.create_grid_map(rows, cols, occupancy=0.3)
        P = zl.create_transition_matrix(M, be_trapped=True)
        x0 = torch.zeros_like(P[0, :])
        x0[0] = 1
        return M, P, x0.reshape(1, -1)

    @unittest.skip
    def test_get_neighbor_indices(self):
        neighbor_indices = zl.get_neighbor_indices(4, 3, 3)
        # expect 1, 3, 5, 7
        print(neighbor_indices == [1, 3, 5, 7])
        neighbor_indices = zl.get_neighbor_indices(0, 3, 3)
        print(neighbor_indices == [1, 3])

    @unittest.skip
    def test_create_grid_map(self):
        M = zl.create_grid_map(10, 10)
        zl.viz_mat(M)
        plt.show()

    @unittest.skip
    def test_create_transition_matrix(self):
        M, P, x0 = self.my_setup()
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        zl.viz_mat(M, ax=ax[0])
        zl.viz_mat(P, ax=ax[1])
        plt.show()

    @unittest.skip
    def test_markov_chan(self):
        M, P, x0 = self.my_setup()
        x = x0
        plt.ion()
        _, ax = plt.subplots(1, 2, figsize=(12, 8))
        plt.suptitle("Markov chain demo")
        ax[0].set_title("The map")
        zl.viz_mat(M, ax[0])
        num_steps = 10000
        num_frames = 20
        for i in range(num_steps):
            x = x @ P
            assert abs(x.sum() - 1.0) < 1e-3, f"x.sum() = {x.sum()}"
            if i % int(num_steps / num_frames) == 0:
                zl.viz_mat(x.reshape(M.size()), ax[1], alpha=0.5)
                plt.title(f"step = {i}")
                plt.pause(0.01)
        plt.ioff()
        plt.show()

    @unittest.skip
    def test_random_walk(self):
        num_rows, num_cols = 30, 30
        M, P, x0 = self.my_setup(num_rows, num_cols, occupancy=0.3)
        x = x0
        num_steps = 1000
        state_prob_history = []
        for i in range(num_steps):
            x = x @ P
            # state_prob_history.append(x.clone().detach())

            # Random sample here
            sample_index = torch.argmax(torch.rand(x.size()) * x)
            # print(x, sample_index)
            x[x > 0.0] = 0
            x[..., sample_index.item()] = 1.0
            state_prob_history.append(x.clone().detach())
        viz_state_prob(M, state_prob_history)

    @unittest.skip
    def test_z_learning(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        num_rows, num_cols = 50, 50
        M, P, x0 = self.my_setup(num_rows, num_cols, occupancy=0.3)
        rho = 0.0
        q = torch.ones_like(x0.squeeze()) * rho
        q[-1] = 0
        zf = torch.zeros_like(x0)
        zf[0, -1] = 1.0
        z = zl.z_learning(P, zf, q, max_num_iters=1e2)
        # c = - torch.log(z) / rho

        num_steps = (num_cols + num_rows) * 2
        Z = torch.diag(z[0, :]).to(device)
        x = x0.clone().detach().to(device)
        P = P.to(device)

        state_prob_history = [x.clone().detach()]
        for i in range(num_steps):
            x = x @ (P @ Z)
            x = x / x.sum()  # normalization
            state_prob_history.append(x.clone().detach())

        viz_state_prob(M, state_prob_history, blit=False, save=False)


if __name__ == '__main__':
    unittest.main()
