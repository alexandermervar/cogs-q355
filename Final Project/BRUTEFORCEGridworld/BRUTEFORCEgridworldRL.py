# To learn how to use reinforcement Q learning I followed this sentdex tutorial
# Link: https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/
# After I followed that tutorial, I then starting to play around with the code

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

size = 10

trainingDuration = 20000
durationInSteps = 200 # How long each episode lasts

movePenalty = 1
enemyPenalty = 300
foodReward = 25
epsilon = 0.9 # set to 0 when loading from a qtable
decay = 0.9999
showEvery = 3000 # How often to show the agent on the screen (in regards to training duration)

start_q_table = None # place fileName here

learningRate = 0.1
discount = 0.95

playerKey = 1  # player key
foodKey = 2  # food key
enemyKey = 3  # enemy key

# Dictionary
d = {1: (0, 255, 0),
     2: (0, 175, 255),
     3: (0, 0, 255)}


class Agent:
    def __init__(self):
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return "Agent Location: [{},{}]".format(self.x, self.y)

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        # Where the actions are defined
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=0, y=1)
        elif choice == 5:
            self.move(x=0, y=-1)
        elif choice == 6:
            self.move(x=-1, y=0)
        elif choice == 7:
            self.move(x=1, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # Out of bounds fix
        if self.x < 0:
            self.x = 0
        elif self.x > size-1:
            self.x = size-1
        if self.y < 0:
            self.y = 0
        elif self.y > size-1:
            self.y = size-1


if start_q_table is None:
    # initialize the q-table
    q_table = {}
    for x1 in range(-size+1, size):
        for y1 in range(-size+1, size):
            for x2 in range(-size+1, size):
                    for y2 in range(-size+1, size):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(8)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(trainingDuration):
    player = Agent()
    food = Agent()
    enemy = Agent()
    if episode % showEvery == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{showEvery} ep mean: {np.mean(episode_rewards[-showEvery:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(durationInSteps):
        obs = (player-food, player-enemy)

        # Retrieves the action
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 8)
        
        player.action(action)

        # TODO: Uncomment to add complexity to model
        # enemy.move()
        # food.move()

        if player.x == enemy.x and player.y == enemy.y:
            reward = -enemyPenalty
        elif player.x == food.x and player.y == food.y:
            reward = foodReward
        else:
            reward = -movePenalty

        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        # Implement Q-learning
        if reward == foodReward:
            new_q = foodReward
        elif reward == -enemyPenalty:
            new_q = -enemyPenalty
        else:
            new_q = (1 - learningRate) * current_q + learningRate * (reward + discount * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((size, size, 3), dtype=np.uint8)
            env[food.x][food.y] = d[foodKey]
            env[player.x][player.y] = d[playerKey]
            env[enemy.x][enemy.y] = d[enemyKey]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((500, 500), resample=Image.BOX)
            cv2.imshow("Reinforcement Learning with Gridworld", np.array(img))
            if reward == foodReward or reward == -enemyPenalty:
                if cv2.waitKey(500) & 0xFF == ord('q'): # wait when episode is over (by either getting food or running into enemy)
                    break
            else:
                if cv2.waitKey(5) & 0xFF == ord('q'): # amount of time to display each step
                    break

        episode_reward += reward
        if reward == foodReward or reward == -enemyPenalty:
            break

    episode_rewards.append(episode_reward)
    epsilon *= decay

moving_avg = np.convolve(episode_rewards, np.ones((showEvery,))/showEvery, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {showEvery}ma")
plt.xlabel("episode #")
plt.show()

with open("myAgentQTable.pickle", "wb") as f:
    pickle.dump(q_table, f)