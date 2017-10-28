import os

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam

from common.gym_runner import GymRunner
from common.q_learning_agent import QLearningAgent


class CartPoleAgent(QLearningAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size, 2000, 'models/cartpole-v0.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=self.state_size))
        model.add(Dense(12, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.action_size))
        model.compile(Adam(lr=0.001), 'mse')
        return model


if __name__ == '__main__':
    gym = GymRunner('CartPole-v0')
    agent = CartPoleAgent(gym.state_size(), gym.action_size())

    gym.train(agent, 1000)
    gym.run(agent, 500)

    agent.save_weights()
    gym.close()
