import random
import gym
from tensorflow.keras import optimizers
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
import os

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

batch_size = 32
n_episodes = 1001

output_dir = 'cartpole'


class DqnAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # gamma = discount factor, how much to discount future reward
        self.gamma = 0.95

        #exploration rate for the agent, instead of exploiting from the knowledge base to make the best possible move, it will help explore further to uncover new possibilities
        # initially epsilon = 1 or 100% as we want to 100% explore in the beginning
        self.epsilon = 1.0

        #However, this epsilon will decay over time
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01 
        
        # step size for stochastic gradient descent optimizer 
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        #Hidden Layer 1 
        model.add(Dense(24, input_dim = self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))

        #output layer
        model.add(Dense(self.action_size, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #what action to take given the state
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_vals = self.model.predict(state)
        return np.argmax(act_vals[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma*np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs = 1, verbose = 0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DqnAgent(state_size, action_size)

done = False
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(5000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            reward = reward
        else:
            reward = -10 
        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state,done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, time, agent.epsilon))
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e%50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
