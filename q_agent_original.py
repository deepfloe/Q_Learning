from q_agent import QAgent
import numpy as np
import gym

class QAgentOriginal(QAgent):
    '''Class to facilitate learning and evaluation of a traditional Q learning agent.
    The rows of the Q table represent states of the environment while columns represent possible actions of the agent.
     The entry in the table for (state,action) is the estimated reward. The training consists of the agent interacting with the environment based on the current Q table. After every training episode the Q table is updated according to the Bellman equation.
    '''
    def __init__(self, env):
        super().__init__(env)
        n_rows = env.observation_space.n
        n_cols = env.action_space.n
        self.Q = np.zeros( (n_rows, n_cols))

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
        # Crucially the q values of the current state is updated depending on the value of the new state
        self.Q[state,action] += learning_rate*(reward+gamma*np.max(self.Q[new_state])-self.Q[state,action])
        return done, new_state

    def greedy_policy(self, state):
        '''Take action which maximises q value (highest expected reward)'''
        action = np.argmax(self.Q[state])
        return action


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    dq = QAgentOriginal(env)
    dq.train(episodes=10000, max_steps=99, learning_rate=0.7, gamma=0.95)
    print(dq.evaluate(max_steps=99, episodes=100))