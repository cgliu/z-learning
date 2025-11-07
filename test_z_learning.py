import pytest

import z_learning as zl

import matplotlib.pyplot as plt
import torch
from random import choices, random

def make_initial_z(num_rows, num_cols):
    z = torch.zeros(1, num_rows * num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            z[0, i * num_cols + j] = num_rows - 1 - i + num_cols - 1 - j
    return torch.exp(-z)

@pytest.fixture
def SetUp():
    def _SetUp(rows = 5, cols = 5, occupancy = 0.3):
        # For pytest.fixture, we can define a factory function to have arguments
        M = zl.create_grid_map(rows, cols, occupancy=occupancy)
        P = zl.create_transition_matrix(M, be_trapped=True)
        x0 = torch.zeros_like(P[0, :])
        x0[0] = 1
        return M, P, x0.reshape(1, -1)
    return _SetUp

@pytest.mark.skip(reason="this test is currently passing")
def test_get_neighbor_indices():
    neighbor_indices = zl.get_neighbor_indices(4, 3, 3)
    # expect 1, 3, 5, 7
    assert (neighbor_indices == [1, 3, 5, 7])
    neighbor_indices = zl.get_neighbor_indices(0, 3, 3)
    assert (neighbor_indices == [1, 3])

@pytest.mark.skip(reason="this test is currently passing")
def test_create_grid_map():
    M = zl.create_grid_map(10, 10)
    zl.viz_mat(M)
    plt.show()

@pytest.mark.skip(reason="this test is currently passing")
def test_create_transition_matrix(SetUp):
    M, P, x0 = SetUp()
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    zl.viz_mat(M, ax=ax[0])
    zl.viz_mat(P, ax=ax[1])
    plt.show()

@pytest.mark.skip(reason="this test is currently passing")
def test_markov_chan(SetUp):
    M, P, x0 = SetUp()
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
    assert True

@pytest.mark.skip(reason="this test is currently passing")
def test_random_walk(SetUp):
    num_rows, num_cols = 30, 30
    M, P, x0 = SetUp(num_rows, num_cols, occupancy=0.3)
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
    zl.viz_state_prob(M, state_prob_history)

@pytest.mark.skip(reason="this test is currently passing")
def test_sto_solver(SetUp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_rows, num_cols = 20, 20
    M, P, x0 = SetUp(num_rows, num_cols, occupancy=0.3)
    rho = 0.0
    q = torch.ones_like(x0.squeeze()) * rho
    q[-1] = 0
    zf = torch.zeros_like(x0)
    zf[0, -1] = 1.0
    z = zl.sto_solver(P, zf, q, max_num_iters=1e2)
    # c = - torch.log(z) / rho

    num_steps = (num_cols + num_rows) * 2
    Z = torch.diag(z[0, :]).to(device)
    x = x0.clone().detach().to(device)
    P = P.to(device)

    state_prob_history = [x.clone().detach()]
    P_star = (P @ Z)
    for i in range(num_steps):
        x = x @ P_star
        x = x / x.sum()  # normalization
        state_prob_history.append(x.clone().detach())

    zl.viz_state_prob(M, state_prob_history, blit=False, save=False)

@pytest.mark.skip(reason="this test is currently passing")
def test_single_rollout(SetUp):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    num_rows, num_cols = 20, 20
    M, P, x0 = SetUp(num_rows, num_cols, occupancy=0.3)
    rho = 0.0
    rollout = zl.single_rollout(x0, P, rho, num_steps=1000)
    zl.viz_state_prob(M, [srs[0] for srs in rollout], blit=True)

@pytest.mark.skip(reason="this test is currently passing")
def test_z_learnging(SetUp):
    num_rows, num_cols = 20, 20
    M, P, x0 = SetUp(num_rows, num_cols, occupancy=0.3)
    rho = 10.0
    x = x0.clone().detach()
    indices = torch.arange(P.size()[0])
    q = torch.ones_like(x0.squeeze()) * rho
    q[-1] = 0
    G = torch.diag(torch.exp(-q).squeeze())

    z = make_initial_z(num_rows, num_cols)

    alpha = 0.9
    def callback(frame_ind):
        nonlocal x, indices, P, G, z, alpha, x0
        i = torch.argmax(x[0, :]).item()
        if i == x.size()[1] - 1:
            x = x0.clone().detach()
            i = 0

        x_next_rand = (x @ P)
        x_next_opt = x_next_rand * z
        #
        x_next = x_next_rand if random() > 0.8 else x_next_opt
        j = choices(indices, x_next[0, :])[0].item()
        z[..., i] = (1-alpha) * z[..., i] + alpha * G[i, i] * z[0, j]

        x = torch.zeros_like(x0)
        x[0, j] = 1.0
        return x.squeeze(), z.squeeze()

    zl.simulation(M, callback, num_steps=100, blit=True, repeat=True, save=False)
