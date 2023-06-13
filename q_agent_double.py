from q_agent import QAgent
import numpy as np
import gym

class QAgentDouble(QAgent):
    '''Class to facilitate learning and evaluation of double Q learning agent.
    The update happens according to the original Hasselt paper.
    '''
    def __init__(self, env):
        super().__init__(env)
        n_rows = env.observation_space.n
        n_cols = env.action_space.n
        #in double Q learning there are two Q tables
        self.Q = np.zeros((2, n_rows, n_cols))

    def update(self, state, action, learning_rate, gamma):
        '''update the double Qtable according to the original Hasselt paper
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
        # i.e if i = 0 then j=1 and vice versa
        action_optimal = np.argmax(q[i, new_state])
        q[i,state,action] += learning_rate*(reward+gamma*q[j,new_state,action_optimal]-q[i,state,action])
        return done, new_state

    def greedy_policy(self, state):
        '''Take action which maximises the average of both Q values'''
        action = np.argmax(self.Q[0,state]+self.Q[1,state])
        return action


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    dq = QAgentDouble(env)
    dq.train(episodes=10000, max_steps=99, learning_rate=0.7, gamma=0.95)
    print(dq.evaluate(max_steps=99, episodes=100))