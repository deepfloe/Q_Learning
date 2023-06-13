import numpy as np
import  gym
from q_agent_original import QAgentOriginal
from q_agent_double import QAgentDouble


class Dice(gym.Env):
    '''Gym implementation of the following game:
The agent rolls a die and sum its number of eyes in each turn. When the agent rolls a six, the game finishes with no reward. If the agent decides to terminate, it gets the current sum as a reward.
    '''
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
def create_baseline_agent(threshold):
    '''Roll while the sum is below the threshold. This strategy is implemented via a Q table, which cannot be interpreted as a table of expected rewards in this case.
    '''
    env = Dice()
    baseline = QAgentOriginal(env)
    for i in range(threshold):
        baseline.Q[i,0] = 1
    for i in range(threshold,100):
        baseline.Q[i,1] = 1

    return baseline


def best_strategy():
    mean_rewards = []
    for i in range(99):
        baseline = create_baseline_agent(i)
        rewards = baseline.evaluate(episodes=1000)
        mean_rewards.append(np.mean(rewards))
    return np.max(mean_rewards)

if __name__ == "__main__":
    env = Dice()
    q = QAgentOriginal(env)
    q.train(max_steps=99, gamma=0.95, learning_rate=0.6, episodes=10000)
    print("Single Q agent:",np.mean(q.evaluate(max_steps=99, episodes = 1000)))

    q = QAgentDouble(env)
    q.train(max_steps=99, gamma=0.95, learning_rate=0.6, episodes=10000)
    print("Double Q agent:", np.mean(q.evaluate(max_steps=99, episodes = 1000)))

    baseline = create_baseline_agent(17)
    print("optimal strategy:",np.mean(baseline.evaluate(max_steps=99, episodes = 1000)))