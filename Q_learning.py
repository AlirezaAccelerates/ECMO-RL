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
# It can be either a trainable or nontrainable model - here I simply put a nueral network

class Approximation_Function():
    def __init__(self):
        
        self.models = []
        for i in range(#number of actions):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([state_], [0]) #Initial state
            self.models.append(model)
            
    def predict(self, s, a=None):
        
        s_ = self.s
        if not a:
            return np.array([m.predict([s_])[0] for m in self.models])
        else:
            return self.models[a].predict([s_])[0]
        
    def update(self, s, a, y):
       
        s_update = self.s
        self.models[a].partial_fit([s_update], [y])
    
    
    
    
l1 = 32                                                        
l2 = 64
l3 = 4


model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
)
    
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
################################################################# Training the RL model
epochs = 1000
losses = [] 
epsilon = 1.0
gamma = 0.9
for i in range(epochs): 
    estimator = Approximation_Function()
    ECMO = OxygenChallengeEnv(#possible initializations)        # Choosing a patient and reset the environments
    state1 = ECMO.states                                        # The initial states for a patient
    status = 1                                                  # Keeping track the processing
    while(status == 1):                                         # Doing ECMO
        qval = estimator.predict(state1)                        # Determining Q value - We should pay attention to the size of the variables
        qval_ = qval.data.numpy()                               # Maybe we do not need this!
        
################################################################# Adding epsilon-greedy method
        if (random.random() < epsilon): 
            action_ = np.random.randint(0,4)                    # Number of possible actions
        else:
            action_ = np.argmax(qval_)
#################################################################
        
        action = action_set[action_]                            #Choosing the best action
        ECMO.DoAction(action) 
        state2 = ECMO.states 
        reward = ECMO.reward()
        
        newQ = estimator.predict(state2)
        td_target = reward + gamma * np.max(newQ)
        maxQ = np.argmax(newQ)                                  # Finding the maximum Q value predicted from the new state
        
        if reward == ? and ! and ...:                           # Is it the last iteration or not? After we define the reward fuction, we should motify this
            Y = reward + (gamma * maxQ)
        else:
            Y = reward
        
        Y = torch.Tensor([Y]).detach()
        X = qval.squeeze()[action_]                            # Createing a copy of the Q value updating the one element corresponding to the action taken
        
        loss = loss_fn(X, Y)
        print(i, loss.item())
        clear_output(wait=True)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        state1 = state2
        
        if reward != ? and ! and ...:                          # Is the game still progressing?
            status = 0
    if epsilon > 0.1:                                          # Decrementing the epsilon value each epoch
        epsilon -= (1/epochs)
        

        
       
