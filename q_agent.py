import numpy as np
import gym
class QAgent:
    '''Abstract super class for different versions of Q learning.
    This class facilitates the training, evaluation and visualization of Q learning tasks in gym.
    The parameter 1-epsilon measures how explorative an agent is, which decays during training. We implement two decay behaviours for epsilon: an exponential decay and an inverse sq decay.
    Attributes:
    :env (gym.Env): a gym environment with discrete action and observation space (currently, tuples of discrete variables are not implemented)
    :training_episodes:
    :n_visits: a numpy that stores how many times a state has been visited. The inverse sq decay depends on that.
   Methods:
    :update: needs to be implemented in subclass
    :greedy_policy: needs to be implemented in subclass
    :epsilon_greedy_policy: pursue greedy policy with probability epsilon
    :train: the agent interacts with the environment according to the epsilon greedy policy and updates the Qtable according to .update() method
    :evaluate: return a list of rewards when the agent interacts with the environment according to greedy policy
    '''
    def __init__(self, env):

        n_rows = env.observation_space.n
        #n_cols = env.action_space.n
        self.env = env
        self.training_episodes = 0
        self.n_visits = np.zeros(n_rows)

    def epsilon_function_inverse_sq(self, state):
        return 1/np.sqrt(self.n_visits[state])

    def epsilon_exp_decay(self, state):
        epsilon_min = 0.05
        epsilon_max = 1
        decay_rate = 0.005
        return np.exp(-self.training_episodes*decay_rate)*(epsilon_max-epsilon_min)+epsilon_min

    def update(self, state, action, learning_rate, gamma):
        '''update the Qtable according the the Bellman equation
        :state (Gym.state): current state of the environment
        :action (Gym.action): action the agent has decided to take
        :param learning_rate(float): between 0 and 1, measures how much quickly it changes the qtable based on new training information
        :gamma(float): parameter that determines the weight of the reward in the new q value
        :returns done (Boolean): whether or not game has terminated
        :returns new_state: new state of environment after action has been performed
        '''
        raise NotImplementedError


    def train(self, episodes, max_steps, learning_rate, gamma):
        '''Run a variable number training episodes to update the Qtable

        :param episodes: number of training episodes
        :param max_steps: the maximum number of steps the agent performs before an episode is aborted
        :param learning_rate, gamma: see update method
        :param decay_rate: determines how quickly epsilon, which measures how likely the agent performs a greedy policy, converges to one
        :param epsilon_min (float): typically near 0, the value epsilon converges to after many episodes
        :param epsilon_max (float): initial value for epsilon, this is only relevant the first time .train() is called.
        '''

        # If epsilon has not been set yet (no training has happened), we set it to the max value
        for _ in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                self.n_visits[state] += 1
                self.training_episodes += 1
                epsilon = self.epsilon_exp_decay(state)
                action = self.epsilon_greedy_policy(state, epsilon)
                done, new_state = self.update(state, action, gamma, learning_rate)
                if done:
                    break
                state = new_state


    def greedy_policy(self, state):
        raise NotImplementedError


    def epsilon_greedy_policy(self,  state, epsilon):
        '''With probabilty epsilon, take a random action, otherwise pursue greedy policy.'''
        x = np.random.rand()
        if x > epsilon:
            action = self.greedy_policy(state)
        else:
            action = self.env.action_space.sample()
        return action


    def evaluate(self, max_steps: int, episodes: int):
        '''Return a list of rewards when agents pursues a greedy policy.
        :param max_steps: maximum number of actions until episode aborts (even if not finished)
        :param episodes: number of evaluation episodes
        :returns: 1D numpy array of awards
        '''
        rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for _ in range(max_steps):
                action = self.greedy_policy(state)
                new_state, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
                state = new_state
            rewards.append(total_reward)
        return np.array(rewards)
