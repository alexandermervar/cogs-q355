import braitenberg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# Graphics Saving Directory
directoryToStoreGraphics = "noTrainGraphics"
parentDirectory = os.getcwd()
path = os.path.join(parentDirectory, "Final Project",directoryToStoreGraphics)
if os.path.isdir(path):
    shutil.rmtree(path)
os.mkdir(path)

# Initialize the Evader and Chaser
chaser = braitenberg.Chaser()
evader = braitenberg.Evader()

# Set duration of simulation
duration = 5000

# Arrays to track the postion of the Evader and Chaser
chaserXPos = np.zeros(duration)
chaserYPos = np.zeros(duration)
evaderXPos = np.zeros(duration)
evaderYPos = np.zeros(duration)

# Simulation Loop
for i in range(duration):
    chaser.readSensors(evader)
    chaser.setWheels()
    chaser.move()
    if chaser.distance(evader) < 0.5:
        chaser.caughtEvader = True
        caughtX = chaser.xpos
        caughtY = chaser.ypos
    evader.setWheels()
    evader.move()
    chaserXPos[i] = chaser.xpos
    chaserYPos[i] = chaser.ypos
    evaderXPos[i] = evader.xpos
    evaderYPos[i] = evader.ypos

# Plot the simualtion
plt.plot(0.0,0.0,"ko")
plt.plot(evader.starting_xpos,evader.starting_ypos,"ko")
if chaser.caughtEvader:
    plt.plot(caughtX, caughtY, "ko")
    plt.text(caughtX-0.5,caughtY+1.0, "Evader Caught")
plt.text(-0.5,1.0,"Chaser Starting Position")
plt.text(evader.starting_xpos-0.5,evader.starting_ypos+1.0,"Evader Starting Position")
plt.scatter(evaderXPos, evaderYPos,s=0.5,c=range(duration), cmap="Blues")
plt.scatter(chaserXPos, chaserYPos,s=0.5, c=range(duration), cmap="Reds")
plt.savefig(path + "/noTrain.png")
plt.show()