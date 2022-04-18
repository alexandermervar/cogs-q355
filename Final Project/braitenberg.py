import random
import numpy as np
import matplotlib.pyplot as plt

class Chaser:
    def __init__(self):
        # Here I initialize all of the variables for the Braitenberg class
        self.xpos = 0.0                                       # Braitenberg vehicle's x position, starts in middle of world
        self.ypos = 0.0                                       # Braitenberg vehicle's y position, starts in middle of world
        self.orientation = np.random.random()*2*np.pi         # Braitenberg vehicle's orientation, starts at random
        self.velocity = 0.0                                   # Braitenberg vehicle's velocity, starts at 0
        self.radius = 1.0                                     # the size/radius of the Braitenberg vehicle
        # The sensor variables are key for location of other vehicle
        self.leftSensor = 0.0                                    # left sensor value
        self.rightSensor = 0.0                                   # right sensor value
        self.leftWheel  = 1.0                                  # left wheel output
        self.rightWheel = 1.0                                  # right wheel output
        
        # Attributes to determine the placement of the sensors
        self.angleoffset = np.pi/2                                                 # left/right sensor angle offset
        self.rs_xpos = self.radius * np.cos(self.orientation + self.angleoffset)   # right sensor x position
        self.rs_ypos = self.radius * np.sin(self.orientation + self.angleoffset)   # right sensor y position
        self.ls_xpos = self.radius * np.cos(self.orientation - self.angleoffset)   # left sensor x position
        self.ls_ypos = self.radius * np.sin(self.orientation - self.angleoffset)   # left sensor y position

        # Chaser Attributes
        self.caughtEvader = False                                                 # boolean to determine if the chaser has caught the evader

    def readSensors(self, evader):
        # This function is what is used to calibrate the sensors in order to track an Evader object

        # Calculate the distance of the Evader for each of the sensors
            self.leftSensor = 1 - np.sqrt((self.ls_xpos-evader.xpos)**2 + (self.ls_ypos-evader.ypos)**2)/10
            self.leftSensor = np.clip(self.leftSensor,0,1)
            self.rightSensor = 1 - np.sqrt((self.rs_xpos-evader.xpos)**2 + (self.rs_ypos-evader.ypos)**2)/10
            self.rightSensor = np.clip(self.rightSensor,0,1)

    def setWheels(self):
        self.rightWheel = random.uniform(0,1)
        self.leftWheel = random.uniform(0,1)

    def move(self):
        # Update the orientation and velocity of the vehicle based on the left and right motors
        self.rightWheel = np.clip(self.rightWheel,0,1)
        self.leftWheel  = np.clip(self.leftWheel,0,1)
        self.orientation += ((self.leftWheel - self.rightWheel)/10) + np.random.normal(0,0.1)
        self.velocity = ((self.rightWheel + self.leftWheel)/2)/50
        
        # Update position of the agent
        self.xpos += self.velocity * np.cos(self.orientation) 
        self.ypos += self.velocity * np.sin(self.orientation)  
        
        # Update position of the sensors
        self.rs_xpos = self.xpos + self.radius * np.cos(self.orientation + self.angleoffset)
        self.rs_ypos = self.ypos + self.radius * np.sin(self.orientation + self.angleoffset)
        self.ls_xpos = self.xpos + self.radius * np.cos(self.orientation - self.angleoffset)
        self.ls_ypos = self.ypos + self.radius * np.sin(self.orientation - self.angleoffset)

    # Calculates the distance of the Chaser from the given Evader object
    def distance(self,evader):
        return np.sqrt((self.xpos-evader.xpos)**2 + (self.ypos-evader.ypos)**2)

class Evader:
    def __init__(self):
        angle = np.random.random()*2*np.pi
        # Here I initialize all of the variables for the Braitenberg class
        self.xpos = 10.0 * np.cos(angle)                      # Braitenberg vehicle's x position, starts in middle of world
        self.ypos = 10.0 * np.sin(angle)                      # Braitenberg vehicle's y position, starts in middle of world
        self.orientation = np.random.random()*2*np.pi         # Braitenberg vehicle's orientation, starts at random
        self.velocity = 0.0                                   # Braitenberg vehicle's velocity, starts at 0
        self.radius = 1.0                                     # the size/radius of the Braitenberg vehicle
        # The sensor variables are key for location of other vehicle
        self.leftWheel  = 0.0                                  # left wheel output
        self.rightWheel = 0.0                                  # right wheel output

    def setWheels(self):
        self.rightWheel = random.uniform(0,1)
        self.leftWheel = random.uniform(0,1)

    def move(self):
        # Update the orientation and velocity of the vehicle based on the left and right motors
        self.rightWheel = np.clip(self.rightWheel,0,1)
        self.leftWheel  = np.clip(self.leftWheel,0,1)
        self.orientation += ((self.leftWheel - self.rightWheel)/10) + np.random.normal(0,0.1)
        self.velocity = ((self.rightWheel + self.leftWheel)/2)/50
        
        # Update position of the agent
        self.xpos += self.velocity * np.cos(self.orientation) 
        self.ypos += self.velocity * np.sin(self.orientation)