import numpy as np

class DQAgent:
    '''
    should make an abstract superclass for the different agents
     '''
    def __init__(self, env):
        n_rows = env.observation_space.n
        n_cols = env.action_space.n
        self.Q = np.zeros((2,n_rows, n_cols))
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

        new_state, reward, done, info = self.env.step(action)
        q = self.Q
        i = np.random.randint(2)
        j = (i+1)%2
        action_optimal = np.argmax(q[i, new_state])
        q[i,state,action] += learning_rate*(reward+gamma*q[j,new_state,action_optimal]-q[i,state,action])
        return done, new_state

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
                #epsilon = self.epsilon_function_inverse_sq(state)
                epsilon = self.epsilon_exp_decay(state)
                action = self.epsilon_greedy_policy(state, epsilon)
                done, new_state = self.update(state, action, gamma, learning_rate)
                if done:
                    break
                state = new_state

    def greedy_policy(self, state):
        '''Take action which maximises q value (highest expected reward)'''
        action = np.argmax(self.Q[0,state]+self.Q[1,state])
        return action

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