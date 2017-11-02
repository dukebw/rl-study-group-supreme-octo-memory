import collections
import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process
import torch
import torch.nn.utils.rnn as rnn_utils


# ======================================================================= #
# Utilities
# ======================================================================= #


def _unpad(padded, seq_lengths):
    """Removes all the padding from a tensor padded along dimension 1, and
    returns the result.
    """
    return torch.cat(
        [padded[i, :seq_len, :] for i, seq_len in enumerate(seq_lengths)],
        dim=0)


class GRULinear(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(GRULinear, self).__init__()
        self.lin_in = torch.nn.Linear(in_features=input_size,
                                      out_features=hidden_size,
                                      bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gru = torch.nn.GRU(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=0.0)
        self.lin_out = torch.nn.Linear(in_features=hidden_size,
                                       out_features=output_size,
                                       bias=True)

    def forward(self, inputs, hidden_prev=None):
        inputs = self.lin_in(inputs)
        inputs = self.relu(inputs)
        outputs, hidden_curr = self.gru(inputs, hidden_prev)

        if isinstance(inputs, rnn_utils.PackedSequence):
            unpacked, seq_lengths = rnn_utils.pad_packed_sequence(
                outputs, batch_first=True)

            # NOTE(brendan): The unpacked sequence is unpadded before being
            # input to the linear layer, so that any values not in the original
            # sequences are thrown away.
            outputs = _unpad(unpacked, seq_lengths)

        pred = self.lin_out(outputs)

        return pred, hidden_curr


def _get_model(obs_dim, action_dim, model_name, num_layers, hidden_size):
    """MLP with four layers of 32 hidden units each."""
    if model_name == 'feedforward':
        layers = []
        for i in range(num_layers):
            prev_size = obs_dim if i == 0 else hidden_size
            out_size = action_dim if i == (num_layers - 1) else hidden_size
            layers.append(('linear{}'.format(i),
                           torch.nn.Linear(in_features=prev_size,
                                           out_features=out_size,
                                           bias=True)))
            if i < (num_layers - 1):
                layers.append(('relu{}'.format(i),
                               torch.nn.ReLU(inplace=True)))

        layers = collections.OrderedDict(layers)
        feedforward = torch.nn.Sequential(layers)
    else:
        feedforward = None

    models = {
        'feedforward': feedforward,
        'gru': GRULinear(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers),
    }

    return models[model_name]


def _init_boxs_loop(obs_dim,
                    action_dim,
                    model_name,
                    is_discrete,
                    num_layers,
                    hidden_size,
                    learning_rate):
    """Initializes Box's loop:
    (model(data) -> inference -> criticism -> updated model(data))*.
    """
    policy = _get_model(obs_dim,
                        action_dim,
                        model_name,
                        num_layers,
                        hidden_size).cuda()

    biases = [p for name, p in policy.named_parameters() if 'bias' in name]
    weights = [p
               for name, p in policy.named_parameters() if 'bias' not in name]

    if is_discrete:
        policy_logstd = None
    else:
        policy_logstd = torch.autograd.Variable(torch.zeros(1).cuda(),
                                                requires_grad=True)
        biases.append(policy_logstd)

    optimizer = torch.optim.Adam([{'params': biases, 'weight_decay': 0},
                                  {'params': weights, 'weight_decay': 1e-4}],
                                 lr=learning_rate)
    # optimizer = torch.optim.Adam(biases + weights, lr=learning_rate)

    return {'policy': policy,
            'policy_logstd': policy_logstd,
            'optimizer': optimizer}


def pathlength(path):
    return len(path["reward"])


def _normal_log_prob(x, means, logstd):
    """Returns the log probability of x under the multi-variate Gaussian
    defined by `means` and `logstd`.

    The Gaussian is assumed to have spherical covariance, so `logstd` is
    expected to be a scalar value.

    The returned log probability is only correct to within an added constant.
    """
    return -0.5*torch.exp(-logstd)*torch.norm(x - means)**2


# ======================================================================= #
# Policy Gradient
# ======================================================================= #

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             num_layers=1,
             size=32,
             model_name='gru'):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # ======================================================================= #
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the
    # numerical values that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for
    # that axis is None
    # ======================================================================= #

    # Observation and action sizes
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

    # ======================================================================= #
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian
    #       distribution over actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces
    #       actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std
    #          'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use
    #          tf.random_normal!)
    #
    #          Should have shape [None, action_dim]
    #
    #      Note: these ops should be functions of the policy network output
    #      ops.
    #
    #   3. Computing the log probability of a set of actions that were actually
    #   taken, according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na',
    #      and the policy network output ops.
    #
    # ======================================================================= #

    boxs_loop = _init_boxs_loop(obs_dim,
                                action_dim,
                                model_name,
                                is_discrete,
                                num_layers,
                                size,
                                learning_rate)
    boxs_loop['policy'].train()

    # ======================================================================= #
    #                           ----------SECTION 5----------
    # Optional Baseline
    # ======================================================================= #

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no,
                                1,
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
        # Define placeholders for targets, a loss function and an update op for
        # fitting a neural network baseline. These will be used to fit the
        # neural network baseline.
        # YOUR_CODE_HERE
        baseline_update_op = TODO

    # ======================================================================= #
    # Training Loop
    # ======================================================================= #

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, actions, rewards = [], [], []
            logprobs = []
            animate_this_episode = ((len(paths) == 0) and
                                    (itr % 10 == 0) and
                                    animate)
            steps = 0
            hidden_curr = None
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)

                ob = torch.autograd.Variable(
                    torch.from_numpy(ob.astype(np.float32))).cuda()
                ob = ob[np.newaxis, np.newaxis, :]
                if model_name == 'feedforward':
                    policy_logits = boxs_loop['policy'](ob)
                else:
                    policy_logits, hidden_curr = boxs_loop['policy'](
                        ob, hidden_curr)
                    policy_logits = policy_logits.squeeze()

                # NOTE(brendan): Here, an action is sampled from the policy.
                # For the discrete case, the policy is interpreted as log
                # probabilities defining a multinomial distribution over the
                # actions.
                #
                # For the continuous case, the policy defines the mean vector
                # for a multi-variate Gaussian with spherical covariance.
                if is_discrete:
                    logprob = torch.nn.LogSoftmax()(policy_logits.unsqueeze(0))
                    logprob = logprob.squeeze()

                    action = torch.multinomial(torch.exp(logprob),
                                               num_samples=1)

                    logprob = logprob[action.data]
                else:
                    if (steps == 0) and (len(paths) == 0):
                        print('logstd {}'.format(
                                boxs_loop['policy_logstd'].data.cpu().numpy()))
                    action = torch.normal(
                        means=policy_logits,
                        std=torch.exp(boxs_loop['policy_logstd']))
                    action = torch.autograd.Variable(action.data)
                    logprob = _normal_log_prob(action,
                                               policy_logits,
                                               boxs_loop['policy_logstd'])
                logprobs.append(logprob)
                action = action.data.cpu().numpy().squeeze()
                actions.append(action)

                ob, rew, done, _ = env.step(action)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break

            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(actions),
                    "logprobs": logprobs}
            paths.append(path)

            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break

        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update
        # by concatenating across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        # =================================================================== #
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be
        # used to compute advantages (which will in turn be fed to the
        # placeholder you defined above).
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t.
        #
        # You will write code for two cases, controlled by the flag
        # 'reward_to_go':
        #
        #   Case 1: trajectory-based PG
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward
        #       summed over entire trajectory (regardless of which time step
        #       the Q-value should be for).
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of
        #       rewards starting from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a
        # variable 'q_n', like the 'ob_no' and 'ac_na' above.
        #
        # =================================================================== #

        q_n = []
        for path in paths:
            if reward_to_go:
                q_path = []
                for t in range(len(path['reward'])):
                    q_for_t = []
                    for steps_from_t, reward in enumerate(path['reward'][t:]):
                        q_for_t.append((gamma**steps_from_t)*reward)

                    q_path.append(np.sum(q_for_t))
            else:
                reward = np.sum([(gamma**t)*r
                                 for t, r in enumerate(path['reward'])])
                q_path = [reward for _ in path['reward']]

            q_n.append(np.array(q_path))

        # =================================================================== #
        #                           ----------SECTION 5----------
        # Computing Baselines
        # =================================================================== #

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict
            # reward-to-go at each timestep for each trajectory, and save the
            # result in a variable 'b_n' like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the
            # statistics (mean and std) of the current or previous batch of
            # Q-values. (Goes with Hint #bl2 below.)

            b_n = TODO
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # =================================================================== #
        #                           ----------SECTION 4----------
        # Advantage Normalization
        # =================================================================== #

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to
            # reduce variance in policy gradient methods: normalize adv_n to
            # have mean zero and std=1.
            all_adv = np.concatenate(adv_n)
            adv_mean = np.mean(all_adv)
            adv_stddev = np.std(all_adv)
            for adv in adv_n:
                adv -= adv_mean
                adv /= adv_stddev + 0.00001

        # =================================================================== #
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        # =================================================================== #
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the
            # inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next
            # iteration. Use the baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly,
            # rescale the targets to have mean zero and std=1. (Goes with Hint
            # #bl1 above.)

            # YOUR_CODE_HERE
            pass

        # =================================================================== #
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        # =================================================================== #

        # Call the update operation necessary to perform the policy gradient
        # update based on the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss
        # function before and after an update, and then log them below.

        loss = torch.autograd.Variable(torch.zeros(1),
                                       requires_grad=True).cuda()
        for i, path in enumerate(paths):
            per_path_loss = 0.0
            for t, logprob in enumerate(path['logprobs']):
                per_path_loss += logprob*adv_n[i][t]
            # TODO(brendan): Does this make sense, averaging reward per
            # timestep?
            loss += per_path_loss/len(path['logprobs'])
        loss = -loss
        loss /= len(paths)
        print(loss.cpu().data.numpy()[0])

        boxs_loop['optimizer'].zero_grad()
        loss.backward()
        for name, param in boxs_loop['policy'].named_parameters():
            print(name, param.size(), param.grad.mean().data.cpu().numpy())
        boxs_loop['optimizer'].step()

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        # logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--num_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                num_layers=args.num_layers,
                size=args.size,
                model_name=args.model_name
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()


if __name__ == "__main__":
    main()
