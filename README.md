# Reinforcement-Learning-DP

## About
Independent project done as part of EECE490W course (Intro to RL) taught at POSTECH. In this assignment, I was required to draw a custom mine-grid world individually with
1.	Different map size.
2.	Determined initial & terminal state position.
3.	Mines are mounted.
4.	States differ within specific probability if action is chosen.

properties and train the agent using reinforcement learning by implementing policy evaluation, value iteration and policy evaluation.

The actions of the agent were as follows:
* If the agent moves left or right, it moves 1 step to that direction with some probability. Otherwise, it moves 2 steps
*	If the agent moves up or down, it moves 1 step to that direction with some probability. Otherwise, it moves 2 steps
* *Wall handling instruction*: When the agent moves to the outside of the world (i.e. the agent meets the wall), it stays still. However, when the agent is moving by 2 steps and falls into the wall, but when it does not fall if the agent proceed to the direction by 1 step, the agent moves 1 step regardless of x or y.
*	*Mine handling instruction*: When the agent passes through a mine but does not finish the action on the mine spot, mine is not exploded.

## Execution
```python girdworld.py```
To return the gridworld with the grid marked accordingly.

```python dp_functions.py```
To run policy evaluation, policy iteration, value iteration to return a number of outputs consisting of reshaped grid policies and value functions.

## Author
Vansh Gupta: https://github.com/V-G-spec

## License
Copyright -2022 - Pohang University of Science and Technology, S. Korea

Part of course EECE490W: Introduction to Reinforcement Learning
