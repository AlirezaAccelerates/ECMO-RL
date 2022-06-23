# This a DQN code with experience replay and target network for ARDS/ECMO project



from Envs import OxygenChallengeEnv                                # We should creat and import our environments
import numpy as np
import torch
import random
from matplotlib import pylab as plt
from collections import deque
import copy

################################################################### Initializing the environment
ECMO = OxygenChallengeEnv(#possible initializations)               # Building up the environments

action_set = {                                                     # Initializing the possible actions
    0: 'Vtest',
    1: 'PEEP',
    2: 'FiO2',
    3: 'Respiratory Rate',
}

################################################################# Creating Deep Q network
    
l1 = 32                                                        
l2 = 64
l3 = 64
l4 = 4


model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),   
)
    
model2 = copy.deepcopy(model) #A
model2.load_state_dict(model.state_dict()) #B

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
  
################################################################# Training the DQN model
epochs = 1000
losses = [] 
epsilon = 1.0
gamma = 0.9
mem_size = 100                                                 # Total size of the experience replay memory
batch_size = 20                                                # The minibatch size of experience replay
replay = deque(maxlen=mem_size)                                # Create the memory replay as a deque list
sync_freq = 10                                                 # Set the update frequency for synchronizing the target model parameters to the main DQN
j = 0
for i in range(epochs):                                         
    ECMO = OxygenChallengeEnv(#possible initializations)        # Choosing a patient
    state_ = ECMO.states + np.random.rand/10                    # The initial states for a patient + a random number to prevent dead neurons
    state1 = torch.from_numpy(state_).float()                   # Converting numpy array to PyTorch variable
    status = 1                                                  # Keeping track the processing
    while(status == 1):                                         # Doing ECMO
        j += 1
        qval = model(state1)                                    # Determining Q value - We should pay attention to the size of variables
        qval_ = qval.data.numpy()
        
################################################################# Adding epsilon-greedy method
        if (random.random() < epsilon): 
            action_ = np.random.randint(0,4)                    # Number of possible actions
        else:
            action_ = np.argmax(qval_)
#################################################################
        
        action = action_set[action_]                            # Choosing the best action
        ECMO.DoAction(action) 
        state2_ = ECMO.states + np.random.rand/10  
        state2 = torch.from_numpy(state2_).float()
        reward = ECMO.reward()
        done = True if reward > 0 else False                    # Is it the last iteration or not?
        exp =  (state1, action_, reward, state2, done) #G
        replay.append(exp)                                      # Add experience to experience replay list
        state1 = state2
      
        if len(replay) > batch_size:                            # Begin minibatch training if replay list is at least as long as minibatch size
            minibatch = random.sample(replay, batch_size)       # Randomly sample a subset of the replay list
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]) 
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
        
           Q1 = model(state1_batch)
            with torch.no_grad():
               Q2 = model2(state2_batch)                       # Use the target network to get the maiximum Q-value for the next state
      
            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0]) 
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
      
            if j % sync_freq == 0:                             # Copy the main model parameters to the target network
               model2.load_state_dict(model.state_dict())

        if reward != ? and ! and ...:                          # Is the game still progressing?
            status = 0
    if epsilon > 0.1:                                          # Or we can define a constant epsilon
        epsilon -= (1/epochs)
