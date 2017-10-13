"""Deep RL Fall 2017 HW1 Section 2."""
import argparse
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def _get_batch(data, batch_indices, start_i, end_i):
    """Get next batch."""
    obs = [data[i, :] for i in batch_indices[start_i:end_i]]
    obs = np.array(obs)
    return torch.from_numpy(obs.astype(np.float32))


def _get_rollout(obs, actions, batch_i, batch_indices, batch_size):
    r"""Get rollout {(x_i, \pi(x_i)}."""
    start_i = batch_i*batch_size
    end_i = (batch_i + 1)*batch_size

    obs = _get_batch(obs, batch_indices, start_i, end_i)
    actions = _get_batch(actions, batch_indices, start_i, end_i)

    return obs, actions

def behaviour_cloning(bhvr_clone_gt_file, batch_size, num_epochs):
    """Imitation learning from saved (obs, actions) pairs from an expert
    policy.
    """
    with open(bhvr_clone_gt_file, 'rb') as f:
        expert_data = pickle.load(f)

    num_train = int(0.7*expert_data['observations'].shape[0])
    obs_train = expert_data['observations'][:num_train, ...]
    obs_val = expert_data['observations'][num_train:, ...]

    actions_train = expert_data['actions'][:num_train, ...]
    actions_val = expert_data['actions'][num_train:, ...]

    model = torch.nn.Linear(
        in_features=expert_data['observations'].shape[-1],
        out_features=expert_data['actions'].shape[-1],
        bias=True)

    criticism = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    plt_train_mean_losses = []
    plt_val_mean_losses = []
    for _ in range(num_epochs):
        model.train()
        batch_indices = list(range(obs_train.shape[0]))
        random.shuffle(batch_indices)

        losses = []
        for batch_i in range(obs_train.shape[0]//batch_size):
            obs, actions = _get_rollout(obs_train,
                                        actions_train,
                                        batch_i,
                                        batch_indices,
                                        batch_size)

            obs = torch.autograd.Variable(obs, requires_grad=True)
            actions = torch.autograd.Variable(actions)

            pred = model(obs)

            mse_loss = criticism(pred, actions)
            losses.append(mse_loss.data[0])

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

        plt_train_mean_losses.append(np.mean(losses))

        model.eval()
        losses = []
        batch_indices = list(range(obs_val.shape[0]))
        for batch_i in range(obs_val.shape[0]//batch_size):
            obs, actions = _get_rollout(obs_val,
                                        actions_val,
                                        batch_i,
                                        batch_indices,
                                        batch_size)

            obs = torch.autograd.Variable(obs, volatile=True)
            actions = torch.autograd.Variable(actions, volatile=True)

            pred = model(obs)

            mse_loss = criticism(pred, actions)
            losses.append(mse_loss.data[0])

        plt_val_mean_losses.append(np.mean(losses))

    epochs = list(range(len(plt_train_mean_losses)))
    plt.plot(epochs, plt_train_mean_losses)
    plt.plot(epochs, plt_val_mean_losses)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhvr-clone-gt-file', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    args = parser.parse_args()

    behaviour_cloning(args.bhvr_clone_gt_file,
                      args.batch_size,
                      args.num_epochs)
