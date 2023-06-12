import numpy as np
import gym
import imageio
import pathlib
class QAgent:
    ''' This class facilitates the training, evaluation and visualization of Q learning tasks in gym.

     The rows of the Q table represent states of the environment while columns represent possible actions of the agent.
     The entry in the table for (state,action) is the estimated reward. The training consists of the agent interacting with the environment based on the current Q table. After every training episode the Q table is updated according to the Bellman equation. One typical policy is to  take the action with the highest expected reward with a probability epsilon and a random action with probability (1-epsilon). The quantity 1-epsilon measures how explorative the agent is and typically decays exponentially while training.
     The better the Q table estimates the rewards the more successful (ie higher reward) interacts with the environment.
    Attributes:
        :env (gym.Env): a gym environment with discrete action and observation space (currently, tuples of discrete variables are not implemented)
        :values (np.array): a numpy array representing the values of the Q table
        :epsilon: a parameter representing how much the agent prefers exploiting over exploring. If epsilon = 0, the agent only exploits( greedy_strategy), if epsilon = 1, the agent acts randomly. During training, epsilon decays.
    Methods:
        :update: update the Qtable according to the Bellman equation
        :train: the agent interacts with the environment according to the epsilon greedy policy and updates the Qtable
        :evaluate: return a list of rewards when the agent interacts with the environment according to greedy policy
        :produce_gif: save a gif image in this files directory of one episode

    Remarks:
        It would make sense to subclass Gym.env, but this class does not have an __init__ method, objects are created by gym.make().
        So, I am not sure what is the best way to subclass, as I cannot call super().__init__
     '''
    def __init__(self, env):
        n_rows = env.observation_space.n
        n_cols = env.action_space.n
        self.values = np.zeros((n_rows, n_cols))
        self.env = env
        self.epsilon = None

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
        q = self.values
        # Crucially the q values of the current state is updated depending on the value of the new state
        q[state,action] += learning_rate*(reward+gamma*np.max(q[new_state])-q[state,action])
        return done, new_state

    def train(self, episodes, max_steps, learning_rate, gamma, decay_rate,  epsilon_min, epsilon_max):
        '''Run a variable number training episodes to update the Qtable

        :param episodes: number of training episodes
        :param max_steps: the maximum number of steps the agent performs before an episode is aborted
        :param learning_rate, gamma: see update method
        :param decay_rate: determines how quickly epsilon, which measures how likely the agent performs a greedy policy, converges to one
        :param epsilon_min (float): typically near 0, the value epsilon converges to after many episodes
        :param epsilon_max (float): initial value for epsilon, this is only relevant the first time .train() is called.
        '''

        # If epsilon has not been set yet (no training has happened), we set it to the max value
        if self.epsilon is None:
            self.epsilon = epsilon_max

        for _ in range(episodes):
            state = self.env.reset()
            for step in range(max_steps):
                action = self.epsilon_greedy_policy(state)
                done, new_state = self.update(state, action, gamma, learning_rate)
                if done:
                    break
                state = new_state
            self.epsilon = np.exp(-decay_rate)*(self.epsilon - epsilon_min) + epsilon_min

    def greedy_policy(self, state):
        '''Take action which maximises q value (highest expected reward)'''
        action = np.argmax(self.values[state])
        return action

    def epsilon_greedy_policy(self,  state):
        '''With probabilty self.epsilon, take a random action, otherwise pursue greedy policy.'''
        x = np.random.rand()
        if x > self.epsilon:
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

    def produce_animation(self, max_steps):
        """Save an mp4 file in the local folder which performs one game with the greedy policy."""
        images = []
        state = self.env.reset()
        img = self.env.render(mode='rgb_array')
        images.append(img)
        for _ in range(max_steps):
            action = self.greedy_policy(state)
            new_state, reward, done, info = self.env.step(action)
            img = self.env.render(mode="rgb_array")
            images.append(img)
            if done:
                break
            state = new_state
        imageio.mimwrite(pathlib.Path("animation.mp4"), images, fps = 1)
        return images

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    q = QAgent(env)
    q.train(max_steps=99, epsilon_max=1.0, epsilon_min=0.05, gamma=0.95, learning_rate= 0.7, episodes=90, decay_rate=0.005)
    print(q.evaluate(max_steps=99, episodes= 1))
    #q.produce_gif(max_steps=99)

    #print(np.mean(test),np.std(test),np.sqrt(1/12))




