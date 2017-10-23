"""Deep RL Fall 2017 HW1 Section 2."""
import argparse
import math
import pickle
import random
import sys

import gym
import load_policy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn.utils.rnn as rnn_utils


class GRULinear(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRULinear, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=0.5)
        self.linear = torch.nn.Linear(in_features=32,
                                      out_features=3,
                                      bias=True)

    def forward(self, inputs, hidden_prev=None):
        outputs, hidden_curr = self.gru(inputs, hidden_prev)

        if isinstance(inputs, rnn_utils.PackedSequence):
            unpacked, seq_lengths = rnn_utils.pad_packed_sequence(
                outputs, batch_first=True)

            # NOTE(brendan): The unpacked sequence is unpadded before being
            # input to the linear layer, so that any values not in the original
            # sequences are thrown away.
            outputs = _unpad(unpacked, seq_lengths)

        pred = self.linear(outputs)

        return pred, hidden_curr


def _unpad(padded, seq_lengths):
    """Removes all the padding from a tensor padded along dimension 1, and
    returns the result.
    """
    return torch.cat(
        [padded[i, :seq_len, :] for i, seq_len in enumerate(seq_lengths)],
        dim=0)


def _get_batch(batch):
    """Gets next batch, returning a padded sequence."""
    seq_lengths = [b.shape[0] for b in batch]

    # NOTE(brendan): Each sequence is padded out to the maximum sequence
    # length.
    for i, seq in enumerate(batch):
        batch[i] = np.pad(
            seq,
            pad_width=((0, seq_lengths[0] - seq.shape[0]), (0, 0)),
            mode='edge')

    batch = np.array(batch)
    batch = batch.astype(np.float32)
    batch = torch.from_numpy(batch)

    return batch, seq_lengths


def _get_rollout(obs, actions, batch_i, batch_indices, batch_size):
    r"""Get rollout {(x_i, \pi(x_i)}.

    Returns a (observations, actions) tuple. `obs` is a packed sequence, where
    all sequences have been padded to the length of the longest sequence in the
    batch.
    """
    def _seq_len(a):
        return a[1].shape[0]

    start_i = batch_i*batch_size
    end_i = (batch_i + 1)*batch_size

    obs = [obs[i] for i in batch_indices[start_i:end_i]]
    obs = sorted(enumerate(obs), key=_seq_len, reverse=True)
    sort_indices = [batch_indices[start_i + o[0]] for o in obs]
    obs = [o[1] for o in obs]

    actions = [actions[i] for i in sort_indices]

    obs, seq_lengths = _get_batch(obs)
    actions, _ = _get_batch(actions)

    obs = torch.autograd.Variable(obs, requires_grad=True).cuda()

    actions = _unpad(actions, seq_lengths)
    actions = torch.autograd.Variable(actions).cuda(async=True)

    obs = rnn_utils.pack_padded_sequence(obs,
                                         lengths=seq_lengths,
                                         batch_first=True)

    return obs, actions


def _evaluate(plt_mean_returns, env, model, num_rollouts):
    """Evaluate mean return of model and store in plt_mean_returns."""
    returns = []
    visited_obs = [[] for _ in range(num_rollouts)]

    model.eval()
    for i in range(10):
        done = False
        hidden_curr = None
        obs = env.reset()
        steps = 0
        total_r = 0.0
        while not done:
            obs = obs[np.newaxis, np.newaxis, :]
            obs = torch.from_numpy(obs.astype(np.float32))
            obs = torch.autograd.Variable(obs, volatile=True)
            action, hidden_curr = model(obs.cuda(), hidden_curr)
            obs, r, done, _ = env.step(action.cpu().data.numpy())
            visited_obs[i].append(obs)
            total_r += r
            steps += 1
            if steps >= env.spec.timestep_limit:
                break

        print('return: {}'.format(total_r))
        returns.append(total_r)

    plt_mean_returns.append(np.mean(returns))

    return visited_obs


def _get_subseqs(subseqs, seq, start_i, subseq_len):
    """Retrieves a set of subsequences from the middle of seq, and appends the
    stacked subsequences to subseqs.
    """
    for i in range((len(seq) - start_i)//subseq_len):
        start = subseq_len*i + start_i
        end = subseq_len*(i + 1) + start_i
        subseqs.append(seq[start:end])


def _form_subsequence_minibatches(subseq_obs, subseq_actions, boxs_loop):
    """Turns lists of observation and action sequences into lists of
    subsequences, with maximum length `subseq_len`.
    """
    subseq_len = 100
    for obs_i, obs_seq in enumerate(boxs_loop['data']['obs']):
        actions_seq = boxs_loop['data']['action'][obs_i]
        if obs_seq.shape[0] < subseq_len:
            subseq_obs.append(obs_seq)
            subseq_actions.append(actions_seq)
            continue

        start_i = math.floor(np.random.uniform(0, subseq_len))
        start_i = min(start_i, obs_seq.shape[0] - subseq_len)

        _get_subseqs(subseq_obs, obs_seq, start_i, subseq_len)
        _get_subseqs(subseq_actions, actions_seq, start_i, subseq_len)


def _train_single_epoch(plt_train_mean_losses, boxs_loop, batch_size):
    """Train model for one epoch."""
    boxs_loop['model'].train()

    subseq_obs = []
    subseq_actions = []
    _form_subsequence_minibatches(subseq_obs, subseq_actions, boxs_loop)

    batch_indices = list(range(len(subseq_obs)))
    random.shuffle(batch_indices)

    losses = []
    for batch_i in range(len(subseq_obs)//batch_size):
        obs, actions = _get_rollout(subseq_obs,
                                    subseq_actions,
                                    batch_i,
                                    batch_indices,
                                    batch_size)

        pred, _ = boxs_loop['model'](obs)

        mse_loss = boxs_loop['criticism'](pred, actions)
        print(mse_loss.cpu().data.numpy()[0])
        losses.append(mse_loss.data[0])

        boxs_loop['optimizer'].zero_grad()
        mse_loss.backward()
        boxs_loop['optimizer'].step()

    plt_train_mean_losses.append(np.mean(losses))


def _get_model(expert_data, model_name):
    """MLP with four layers of 32 hidden units each."""
    models = {
        'feedforward': torch.nn.Sequential(
            torch.nn.Linear(
                in_features=expert_data['observations'][0].shape[-1],
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
                out_features=expert_data['actions'][0].shape[-1],
                bias=True)),
        'gru': GRULinear(
            input_size=expert_data['observations'][0].shape[-1],
            hidden_size=32,
            num_layers=3),
    }

    return models[model_name]


def _init_boxs_loop(expert_data, model_name):
    """Initializes Box's loop:
    (model(data) -> inference -> criticism -> updated model(data))*.
    """
    model = _get_model(expert_data, model_name).cuda()

    criticism = torch.nn.MSELoss().cuda()

    biases = [p for name, p in model.named_parameters() if 'bias' in name]
    weights = [p for name, p in model.named_parameters() if 'bias' not in name]
    optimizer = torch.optim.Adam([{'params': biases, 'weight_decay': 0},
                                  {'params': weights, 'weight_decay': 1e-4}],
                                 lr=1e-3)

    return {'data': {'obs': expert_data['observations'],
                     'action': expert_data['actions']},
            'model': model,
            'criticism': criticism,
            'optimizer': optimizer}


def _subplot(nrows_ncols_plotnum, x, y, xlab, ylab):
    """Create subplot using the nrows_ncols_plotnum shorthand."""
    ax = plt.subplot(nrows_ncols_plotnum)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    plt.plot(x, y)


def behaviour_cloning(flags):
    """Imitation learning from saved (obs, actions) pairs from an expert
    policy.
    """
    torch.backends.cudnn.benchmark = True

    env = gym.make(flags.envname)

    # NOTE(brendan): expert_data is expected to be a list of numpy arrays of
    # observations/actions. Each outer list represents a sequence constituting
    # one episode observed by the expert.
    with open(flags.bhvr_clone_gt_file, 'rb') as f:
        expert_data = pickle.load(f)

    policy_fn = load_policy.load_policy(flags.expert_policy_file)

    boxs_loop = _init_boxs_loop(expert_data, flags.model_name)

    plt_train_mean_losses = []
    plt_mean_returns = []
    for _ in range(flags.num_epochs):
        _train_single_epoch(plt_train_mean_losses, boxs_loop, flags.batch_size)

        visited_obs = _evaluate(plt_mean_returns,
                                env,
                                boxs_loop['model'],
                                flags.num_rollouts)

        expert_actions = [[] for _ in range(flags.num_rollouts)]
        for i, rollout in enumerate(visited_obs):
            for obs in rollout:
                expert_actions[i].append(policy_fn(obs[None, :]))

        expert_actions = [np.squeeze(np.array(a, dtype=np.float32))
                          for a in expert_actions]
        visited_obs = [np.squeeze(np.array(o, dtype=np.float64))
                       for o in visited_obs]

        boxs_loop['data']['obs'] += visited_obs
        boxs_loop['data']['action'] += expert_actions

        sys.stdout.flush()

    epochs = list(range(len(plt_train_mean_losses)))

    plt.figure(1)
    _subplot(211, epochs, plt_train_mean_losses, 'Epoch', 'Training loss')
    _subplot(212, epochs, plt_mean_returns, 'Epoch', 'Return')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhvr-clone-gt-file', type=str, default=None)
    parser.add_argument('--envname', type=str, default=None)
    parser.add_argument('--expert-policy-file', type=str, default=None)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--num-rollouts', type=int, default=None)
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)):
        behaviour_cloning(args)
