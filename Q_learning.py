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
    ECMO = OxygenChallengeEnv(#possible initializations)        # Choosing a patient
    state_ = ECMO.states                                        # The initial state for a patient
    state1 = torch.from_numpy(state_).float()                   # Converting numpy array to PyTorch variable
    status = 1                                                  # Keeping track the processing
    while(status == 1):                                         # Doing ECMO
        qval = model(state1)                                    # Determining Q value - We should pay attention to the size of variables
        qval_ = qval.data.numpy()
        
################################################################# Adding epsilon-greedy method
        if (random.random() < epsilon): 
            action_ = np.random.randint(0,4)                    # Number of possible actions
        else:
            action_ = np.argmax(qval_)
#################################################################
        
        action = action_set[action_]                            #Choosing the best action
        ECMO.DoAction(action) 
        state2_ = ECMO.states 
        state2 = torch.from_numpy(state2_).float()
        reward = ECMO.reward()
        
        with torch.no_grad():                                   # As we don't want any learning here. Just updating
            newQ = model(state2)
        maxQ = torch.max(newQ)                                  # Finding the maximum Q value predicted from the new state
        
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
        

        
       
