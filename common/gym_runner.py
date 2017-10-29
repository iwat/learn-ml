import gym
from gym import wrappers


class GymRunner:
    def __init__(self, env_id, max_timesteps=100000):
        self.max_timesteps = max_timesteps

        self.env = gym.make(env_id)

    def state_size(self):
        return self.env.observation_space.shape[0]

    def action_size(self):
        return self.env.action_space.n

    def calc_reward(self, state, action, gym_reward, next_state, done):
        return gym_reward

    def train(self, agent, num_episodes, autosave=True):
        self.run(agent, num_episodes, autosave=True, do_train=True)

    def run(self, agent, num_episodes, autosave=True, do_train=False):
        for episode in range(num_episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            total_reward = 0

            for t in range(self.max_timesteps):
                action = agent.select_action(state, do_train)

                # execute the selected action
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                reward = self.calc_reward(state, action, reward, next_state, done)

                # record the results of the step
                if do_train:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                if done:
                    break

            # train the agent based on a sample of past experiences
            if do_train:
                agent.replay()
                if episode % 10 == 0:
                    agent.save_weights()

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, num_episodes, total_reward, agent.epsilon if do_train else 0))

    def close(self):
        self.env.close()
