# RL_A3C
Asynchronous Advantage Actor Critic implementation using Tensorflow for OpenAI-Gym (ATARI) environments.

Based on the [A3C paper]()

## Requirements

* Python 3
* OpenAI-Gym
* Tensorflow
* scipy
* numpy

## Usage

You can run the code simply by using:

```
python a3c.py
```

To change the environment and the number of threads just change the following line:

```
def main():
    a3c("Breakout-v0", num_threads=8 )
```

This implementation takes an **image** input and has an output for a **discrete** action space. All ATARI2600 games from OpenAI-Gym should work. If you'd like to use a **continuous** input (not images), you'd have to change the first two layers of the model in `agent.py` and the preprocessing for the observation in `custom_gym.py`.

In order to change learning parameters go to these lines:

```
T_MAX = 10000000 # maximum steps
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
TEST_EVERY = 30000
I_ASYNC_UPDATE = 5 #horizon for an update
```
For learning purposes I encourage you to test different parameters and compare performances.

Unfortunately most comments are in Portuguese-BR, I'll be working on translating them as soon as possible.

## To add:
- [ ] Translate comments
- [ ] Continuous action
- [ ] LSTM

## Useful links:

[Theoretical explaination on Policy Gradients](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)

[Pieter Abbeel's lecture on policy gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44)

[Chris Nicholls' very simple A3C tutorial](https://cgnicholls.github.io/reinforcement-learning/2017/03/27/a3c.html) - A lot of my code is based on this.

[Arthur Juliani's A3C tutorial](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

[Morvan Zhou's implementation of A3C](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/10_A3C) - I suggest you have a basic understanding of the A3C before diving into his code.
