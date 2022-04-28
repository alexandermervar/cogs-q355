import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque

import time

import numpy as np

# NOTE: You can use the _ to denote a comma in a large number in python in order to make it easier to read
# This is used with the batch size. So, we take a random sample of this size to act as the batch.

# Therefore, we have stability in the training method with the two models but, we also stability in the predictions
# due to the fact that we are not overfitting to any one particular observation. Instead, we are updating the weights
# in accordance to a batch of observations.
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
# Size of sample from memory for training
MINIBATCH_SIZE = 64

# Define the model name
MODEL_NAME = "256x2"

# TODO: Leave a comment to describe what this is
DISCOUNT = 0.99

# This class creates the Deep Q-Learning model
class DQNAgent:

    def __init__(self):

        # Define two models, here. 
        # This allows the model to have some consistency between iterations
        # Then after a certain number of iterations, n, we can then updated the second model.

        # The model that we are training
        self.model = self.createModel()

        # The model that we are using to predict the next action
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())

        # See the comment above where I define the replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # TODO: Figure out what the use of this is.
        # NOTE: I think this is used to save the weights of the model
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # TODO: Leave a comment to explain these index values (i.e. transition[0], transition[3])
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(x)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)







# ======================================================================================================================

# Totally stole this from a Sentdex tutorial: 
# https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)