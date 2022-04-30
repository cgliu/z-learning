import unittest

import z_learning as zl

import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from random import choices


class TestZLearning(unittest.TestCase):
    def my_setup(self, rows=5, cols=5):
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
        M, P, x0 = self.my_setup()
        x = x0
        plt.ion()
        _, ax = plt.subplots(figsize=(10, 8))
        plt.suptitle("Markov chain demo")
        ax.set_title("The map")
        zl.viz_mat(M, ax)
        num_steps = 100
        num_frames = 100
        for i in range(num_steps):
            x = x @ P
            # random sample here
            sample_index = torch.argmax(torch.rand(x.size()) * x)
            print(x, sample_index)
            x[x > 0.0] = 0
            x[..., sample_index.item()] = 1.0
            if i % int(num_steps / num_frames) == 0:
                zl.viz_mat(x.reshape(M.size()) * 0.5, ax, alpha=0.5)
                zl.viz_mat(M, ax, alpha=0.5)
                plt.title(f"step = {i}")
                plt.pause(0.01)
        plt.ioff()
        plt.show()

    # @unittest.skip
    def test_z_learning(self):
        num_rows, num_cols = 20, 20

        M, P, x0 = self.my_setup(num_rows, num_cols)
        rho = 0.5
        q = torch.ones_like(x0.squeeze()) * rho
        q[-1] = 0
        zf = torch.zeros_like(x0)
        zf[0, -1] = 1.0
        z = zl.z_learning(P, zf, q, max_num_iters=1e4)
        c = - torch.log(z) / rho

        num_steps = 50
        num_frames = 50
        Z = torch.diag(z[0, :])
        x = x0.clone().detach()
        state_ind_history = [0]
        for i in range(num_steps):
            x = x @ (P @ Z)
            sample_index = choices(list(range(x.size()[1])), x[0, :])[0]
            #  Collapse probability
            x[x > 0.0] = 0
            x[..., sample_index] = 1.0
            #
            state_ind_history.append(sample_index)

        fig, ax = plt.subplots(figsize=(10, 8))
        zl.viz_mat(M, ax, alpha=0.5)
        patch = plt.Rectangle((0.5, 0.5),
                              1, 1,
                              fc='red',
                              ec='red',
                              lw=1)

        def init():
            ax.add_patch(patch)
            return patch,

        def animate(i):
            state_ind = state_ind_history[i]
            row = int(state_ind / num_cols)
            col = state_ind - row * 20
            patch.set_xy((col, row))
            return patch,

        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=num_steps,
                                       interval=100,
                                       blit=True)
        # anim.save("anim.gif", writer="imagemagick")
        plt.show()


if __name__ == '__main__':
    unittest.main()
