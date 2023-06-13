# Q-Learning in Gym
Q learning is one of the simplest techniques in reinforcement learning. It is useful when the space of states and actions is discrete and relatively small. In the python library `gym`, such environments are already implemented and called [toy text environments](https://www.gymlibrary.dev/environments/toy_text/).
We implement a class to facilitate training, evaluation and visualizations for Q learning tasks in gym. 

## How to use
You need to install the package gym. The file `q_agent.py` contains the base class for Q learning. Currently, there are two subclasses available: original Q learning in the file `q_agent_original.py` and double Q learning in `q_agent_double.py` as in [Hasselt](https://papers.nips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf).
In general proceed as follows
1. Define a gym environment with `gym.make()`
2. Create a QAgent instance
3. Use the QAgent methods `train()`, `evaluate()`

Here's an example code for Cliff Walking in `q_agent_original.py`:
```python
    env = gym.make('CliffWalking-v0')
    q = QAgentOriginal(env)
    q.train(episodes=10000, max_steps=99, learning_rate=0.7, gamma=0.95)
    print(q.evaluate(max_steps=99, episodes=100))
```
## Dice Game
The repo also contains an implementation of a custom Gym environment called `dice.py`. The agent rolls a die and sum its number of eyes in each turn. When the agent rolls a six, the game finishes with no reward. If the agent decides to terminate, it gets the current sum as a reward. This is an example where the reward is not deterministic, a setting where classical Q-learning is known to overerstimate Q-values. We see that double Q learning performs better than original Q leanring but is still suboptimal.
 
