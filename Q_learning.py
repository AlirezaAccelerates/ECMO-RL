# This a Q_learning code for ARDS/ECMO project



from Envs import OxygenChallengeEnv                                # We should creat and import our environments
import numpy as np
import torch
import random
from matplotlib import pylab as plt

################################################################### Initializing the environment
ECMO = OxygenChallengeEnv(#possible initializations)               # Building up the environments

action_set = {                                                     # Initializing the possible actions
    0: 'Vtest',
    1: 'PEEP',
    2: 'FiO2',
    3: 'Respiratory Rate',
}

################################################################# Creating Q function
# I put a linear approximation function as the Q_learning model
from sklearn.linear_model import SGDRegressor
    
class Approximation_Function():
    def __init__(self):
        
        self.models = []
        for i in range(#number of actions):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([state_], [0]) #Initial state
            self.models.append(model)
            
    def predict(self, s, a=None):                                 # Predicting Q value
        
        s_ = self.s
        if not a:
            return np.array([m.predict([s_])[0] for m in self.models])
        else:
            return self.models[a].predict([s_])[0]
        
    def update(self, s, a, y):
       
        s_update = self.s
        self.models[a].partial_fit([s_update], [y])
    

################################################################# Training the RL model
epochs = 1000
epsilon = 1.0
gamma = 0.9
for i in range(epochs): 
    estimator = Approximation_Function()
    ECMO = OxygenChallengeEnv(#possible initializations)        # Choosing a patient and reset the environments
    state1 = ECMO.states                                        # The initial states for a patient
    status = 1                                                  # Keeping track the processing
    while(status == 1):                                         # Doing ECMO
        qvals = estimator.predict(state1)                       # Determining Q value - We should pay attention to the size of the variables
        qvals_ = qval.data.numpy()                              # Maybe we do not need this!
        
################################################################# Adding epsilon-greedy method
        if (random.random() < epsilon): 
            action_ = np.random.randint(0,4)                    # Number of possible actions
        else:
            action_ = np.argmax(qvals_)
#################################################################
        
        action = action_set[action_]                            # Choosing the best action
        ECMO.DoAction(action) 
        state2 = ECMO.states 
        reward = ECMO.reward()
        
        new_qvals = estimator.predict(state2)                   # Finding the maximum Q value predicted from the new state
        td_target = reward + gamma * np.max(new_qvals)                             
        
        print("\rEpisode {} ({})".format(i , reward))

        state1 = state2
        
        estimator.update(state1, action, td_target)            # Temporal Difference Algorithms
        
        
        if reward != ? and ! and ...:                          # Is the ECMO still progressing?
            status = 0
    if epsilon > 0.1:                                          # Decrementing the epsilon value in each epoch
        epsilon -= (1/epochs)
        

        
       
