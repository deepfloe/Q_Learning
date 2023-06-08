# Q-Learning in Gym
Q learning is one of the simplest techniques in reinforcement learning. It is useful when the space of states and actions is discrete and relatively small. In the python library `gym`, such environments are already implemented and called [toy text environments](https://www.gymlibrary.dev/environments/toy_text/).
We implement a class to facilitate training, evaluation and visualizations for Q learning tasks in gym. Some of the code is taken from [this tutorial](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial).

## How to use
You need to install the packages gym and imageio for visualization
In the file `training_pipeline.py`, proceed as follows
1. Define a gym environment with `gym.make()`
2. Create a Qtable instance
3. Use the Qtable methods `train()`, `evaluate()` and `produce_animation()`

Here's an example code for Cliff Walking:
```python
    env = gym.make('CliffWalking-v0')
    q = Qtable(env)
    q.train(max_steps=99, epsilon_max=1.0, epsilon_min=0.05, gamma=0.95,
            learning_rate= 0.7, episodes=90, decay_rate=0.005)
    print(q.evaluate(max_steps=99, episodes= 1))
    q.produce_gif(max_steps=99)
```
## Next Steps
1. Generalize the class so it can handle tuples of Discrete spaces
2. Implement a custom gym environment and use the training pipeline on it
## Current issues
It would make sense to make Qtable a subclass of `Gym.env`, but this class does not have a constructor method, objects are created by `gym.make()`. So, I am not sure what is the best way to subclass, as I cannot call `super().__init__()`.
