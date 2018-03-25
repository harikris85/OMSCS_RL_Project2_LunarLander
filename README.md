# OMSCS_RL_Project2_LunarLander
Deep Reinforcement Learning with Double Q Network (DDQN) based approach to solve Open AI Gym open Ai Lunar Lander Problem
https://gym.openai.com/envs/LunarLander-v2/

# Problem 
# Description 
For this project, you will be writing an agent to successfully land the “Lunar Lander” that is  implemented in OpenAI gym.  You are free to use and extend any type of RL agent discussed in  this class. 
# Lunar Lander Environment 
The problem consists of a 8-dimensional continuous state space and a discrete action space.  There are four discrete actions available: do nothing, fire the left orientation engine, fire the main  engine, fire the right orientation engine. The landing pad is always at coordinates (0,0).  Coordinates consist of the first two numbers in the state vector. The total reward for moving from  the top of the screen to landing pad ranges from 100 - 140 points varying on lander placement on  the pad. If lander moves away from landing pad it is penalized the amount of reward that would  be gained by moving towards the pad. An episode finishes if the lander crashes or comes to rest,  receiving additional -100 or +100 points respectively. Each leg ground contact is worth +10 points.  Firing main engine incurs a -0.3 point penalty for each occurrence. Landing outside of the landing  pad is possible. Fuel is infinite, so, an agent could learn to fly and then land on its first attempt.  The problem is considered solved when achieving a score of 200 points or higher. 
