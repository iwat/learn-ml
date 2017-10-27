import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gym_runner import GymRunner
from q_learning_agent import QLearningAgent


class CartPoleAgent(QLearningAgent):
    def __init__(self):
        QLearningAgent.__init__(self, 4, 2)

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/cartpole-v0.h5")
        return model


if __name__ == "__main__":
    gym = GymRunner('CartPole-v0')
    agent = CartPoleAgent()

    gym.train(agent, 1000)
    gym.run(agent, 500)

    agent.model.save_weights("models/cartpole-v0.h5", overwrite=True)
    gym.close
