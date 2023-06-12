import numpy as np
import  gym
from doubleQ import DQAgent
class Dice(gym.Env):
    def __init__(self, render_mode =None):

        self.observation_space = gym.spaces.Discrete(100)
        # 0 = roll again, 1 = stay
        self.action_space = gym.spaces.Discrete(2)

    def sample(self):
        return self.action_space.sample(), self.observation_space.sample()

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.sum = 0
        #info = {}
        return self.sum

    def step(self, action):
        if action == 0:
            reward = 0
            dice = np.random.randint(6)+1
            if dice == 6:
                done = True
                self.sum = 0
            else:
                done = False
                self.sum+=dice

        if action == 1:
            reward = self.sum
            done = True
        info = {}

        return self.sum ,reward, done, info

    def render(self):
        return

class Baseline:
    def __init__(self, threshold):
        self.sum = 0
        self.threshold = threshold

    def roll_again(self):
        dice = np.random.randint(low=1, high= 7)
        if dice == 6:
            done = True
            self.sum = 0
        else:
            done = False
            self.sum += dice
        return done

    def reset(self):
        self.sum = 0

    def play(self):
        self.reset()
        while self.sum<self.threshold:
            if self.roll_again():
                break
        return self.sum

    def evaluate(self, episodes):
        return np.array([self.play() for _ in range(episodes)])

def best_strategy():
    mean_rewards = []
    for i in range(99):
        env2 = Baseline(i)
        rewards = env2.evaluate(episodes=1000)
        mean_rewards.append(np.mean(rewards))
    return np.max(mean_rewards)

#env = Dice()
#q = QAgent(env)
#q.train(max_steps=99, epsilon_max=1.0, epsilon_min=0.05, gamma=0.95, learning_rate=0.7, episodes=10000, decay_rate=0.005)
#print(np.mean(q.evaluate(max_steps=99, episodes = 1000)))
env = Dice()
q = DQAgent(env)
q.train(max_steps=99, gamma=0.95, learning_rate=0.6, episodes=10000)
print(np.mean(q.evaluate(max_steps=99, episodes = 1000)))



#print(best_strategy())

#print(np.mean(env2.evaluate(episodes=100)))