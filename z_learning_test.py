import unittest

import z_learning as zl
import matplotlib.pyplot as plt
import torch


class TestZLearning(unittest.TestCase):
    def my_setup(self):
        M = zl.create_grid_map(10, 10)
        P = zl.create_transition_matrix(M, be_trapped=True)
        x0 = torch.zeros_like(P[0, :])
        x0[0] = 1
        return M, P, x0.reshape(1, -1)

    def test_get_neighbor_indices(self):
        neighbor_indices = zl.get_neighbor_indices(4, 3, 3)
        # expect 1, 3, 5, 7
        print(neighbor_indices == [1, 3, 5, 7])
        neighbor_indices = zl.get_neighbor_indices(0, 3, 3)
        print(neighbor_indices == [1, 3])

    # def test_create_grid_map(self):
    #     M = zl.create_grid_map(10, 10)
    #     zl.viz_mat(M)
    #     plt.show()

    # def test_create_transition_matrix(self):
    #     M = zl.create_grid_map(5, 5)
    #     P = zl.create_transition_matrix(M)
    #     _, ax = plt.subplots(1, 2, figsize=(12, 6))
    #     zl.viz_mat(M, ax=ax[0])
    #     zl.viz_mat(P, ax=ax[1])
    #     plt.show()

    @unittest.skip
    def test_markov_chan(self):
        # M = zl.create_grid_map(20, 20)
        # P = zl.create_transition_matrix(M, be_trapped=False)
        # x0 = torch.zeros_like(P[0, :])
        # x0[0] = 1
        # x = x0.reshape(1, -1)
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

    def test_z_learning(self):
        M, P, x0 = self.my_setup()
        rho = 2

        # q = torch.ones_like(x0.squeeze()) * rho
        # q[-1] = 0
        q = torch.exp(-M.flatten() * rho)
        q[-1] = 0

        # print(q.size())
        zf = torch.zeros_like(x0)
        zf[0, -1] = 1.0
        # print(zf)

        z = zl.z_learning(P, zf, q, max_num_iters=1e3)
        c = - torch.log(z) / rho
        _, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].set_title("The map")
        zl.viz_mat(M, ax[0])
        zl.viz_mat(c.reshape(M.size()), ax[1], annot=True)
        plt.show()


if __name__ == '__main__':
    unittest.main()
