"""OpenAI gym tutorial."""
import gym


def min_env():
    """Minimal environment example."""
    env = gym.make('CartPole-v0')
    env.reset()

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            print(action)
            observation, _, is_done, _ = env.step(action)
            print(observation)
            if is_done:
                print('Episode {} finished after {} timesteps.'
                      .format(i_episode, t))
                break


if __name__ == '__main__':
    min_env()
