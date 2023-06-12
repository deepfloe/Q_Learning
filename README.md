# Q-Learning in Gym
Q learning is one of the simplest techniques in reinforcement learning. It is useful when the space of states and actions is discrete and relatively small. In the python library `gym`, such environments are already implemented and called [toy text environments](https://www.gymlibrary.dev/environments/toy_text/).
We implement a class to facilitate training, evaluation and visualizations for Q learning tasks in gym. Some of the code is taken from [this tutorial](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial).

## How to use
You need to install the packages gym and imageio for visualization
In the file `training_pipeline.py`, proceed as follows
1. Define a gym environment with `gym.make()`
2. Create a QAgent instance
3. Use the QAgent methods `train()`, `evaluate()` and `produce_animation()`

Here's an example code for Cliff Walking:
```python
    env = gym.make('CliffWalking-v0')
    q = QAgent(env)
    q.train(max_steps=99, epsilon_max=1.0, epsilon_min=0.05, gamma=0.95,
            learning_rate= 0.7, episodes=90, decay_rate=0.005)
    print(q.evaluate(max_steps=99, episodes= 1))
    q.produce_gif(max_steps=99)
```
### Dice Game and double Q learning
The repo also contains an implementation of a custom Gym environment called `dice.py`. The agent rolls a die and sums its number. When the agent rolls a six, the game finishes with no reward. If the agent decides to terminate, it gets the current sum as a reward. This is an example where the reward is not deterministic, a setting where classical Q-learning is known to overerstimate Q-values.

The file `doubleQ.py` implements the original double Q learning algorithm from the original [Hasselt](https://papers.nips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf) paper.


