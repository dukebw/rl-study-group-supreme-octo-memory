"""Deep RL Fall 2017 HW1 Section 2."""
import argparse
import pickle
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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


def _evaluate(plt_mean_returns, env, model):
    """Evaluate mean return of model and store in plt_mean_returns."""
    model.eval()
    returns = []
    obs = env.reset()
    for _ in range(10):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            obs = torch.from_numpy(obs.astype(np.float32))
            obs = torch.autograd.Variable(obs, volatile=True)
            action = model(obs.cuda())
            obs, r, done, _ = env.step(action.cpu().data.numpy())
            total_r += r
            steps += 1
            if steps >= env.spec.timestep_limit:
                break

        returns.append(total_r)

    plt_mean_returns.append(np.mean(returns))


def _train_single_epoch(plt_train_mean_losses,
                        boxs_loop,
                        batch_indices,
                        batch_size):
    """Train model for one epoch."""
    random.shuffle(batch_indices)
    boxs_loop['model'].train()

    losses = []
    for batch_i in range(boxs_loop['data']['obs'].shape[0]//batch_size):
        obs, actions = _get_rollout(boxs_loop['data']['obs'],
                                    boxs_loop['data']['action'],
                                    batch_i,
                                    batch_indices,
                                    batch_size)

        obs = torch.autograd.Variable(obs, requires_grad=True).cuda()
        actions = torch.autograd.Variable(actions).cuda(async=True)

        pred = boxs_loop['model'](obs)

        mse_loss = boxs_loop['criticism'](pred, actions)
        losses.append(mse_loss.data[0])

        boxs_loop['optimizer'].zero_grad()
        mse_loss.backward()
        boxs_loop['optimizer'].step()

    plt_train_mean_losses.append(np.mean(losses))


def _get_model(expert_data):
    """MLP with four layers of 32 hidden units each."""
    return torch.nn.Sequential(
        torch.nn.Linear(
            in_features=expert_data['observations'].shape[-1],
            out_features=32,
            bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=32,
            out_features=32,
            bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=32,
            out_features=32,
            bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=32,
            out_features=expert_data['actions'].shape[-1],
            bias=True),
        )


def behaviour_cloning(bhvr_clone_gt_file, batch_size, num_epochs, envname):
    """Imitation learning from saved (obs, actions) pairs from an expert
    policy.
    """
    torch.backends.cudnn.benchmark = True

    env = gym.make(envname)

    with open(bhvr_clone_gt_file, 'rb') as f:
        expert_data = pickle.load(f)

    obs_train = expert_data['observations']
    actions_train = expert_data['actions']

    model = _get_model(expert_data).cuda()

    criticism = torch.nn.MSELoss().cuda()

    biases = [p for name, p in model.named_parameters() if 'bias' in name]
    weights = [p for name, p in model.named_parameters() if 'bias' not in name]
    optimizer = torch.optim.Adam([{'params': biases, 'weight_decay': 0},
                                  {'params': weights, 'weight_decay': 1e-4}],
                                 lr=1e-3)

    boxs_loop = {'data': {'obs': obs_train, 'action': actions_train},
                 'model': model,
                 'criticism': criticism,
                 'optimizer': optimizer}

    plt_train_mean_losses = []
    plt_mean_returns = []
    for _ in range(num_epochs):
        batch_indices = list(range(obs_train.shape[0]))

        _train_single_epoch(plt_train_mean_losses,
                            boxs_loop,
                            batch_indices,
                            batch_size)

        _evaluate(plt_mean_returns, env, model)

    epochs = list(range(len(plt_train_mean_losses)))

    plt.figure(1)
    ax = plt.subplot(211)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')

    plt.plot(epochs, plt_train_mean_losses)

    ax = plt.subplot(212)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Return')

    plt.plot(epochs, plt_mean_returns)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhvr-clone-gt-file', type=str, default=None)
    parser.add_argument('--envname', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)):
        behaviour_cloning(args.bhvr_clone_gt_file,
                          args.batch_size,
                          args.num_epochs,
                          args.envname)
