import torch
from torch import nn
import random
import gymnasium as gym
from tqdm import trange
import torch.optim as optim
import matplotlib.pyplot as plt


from MyEnv import MyEnv

env=MyEnv(gym.make("ErgoGripper-Square-Touch-Double-Graphical-v1",headless=True))

GAMMA = 0.9
loss_function = nn.MSELoss()

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()

        self.fc1=nn.Linear(6,16)
        self.fc2=nn.Linear(16,9)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)

        return x
    
neural_network=NN()
optimizer = optim.Adam(neural_network.parameters(), lr=0.0001)

def select_action(state,episode):
    if episode<50:
        epsilon=1
    else:
        epsilon=1-(episode-50)/950
    predicted_q=neural_network(state)
    if random.random()<epsilon or episode<10:
        return torch.max(predicted_q),torch.tensor(random.randrange(9))
    else:
        return torch.max(predicted_q),torch.argmax(predicted_q)

        
losses=[]
mean_rewards=[]

for episode in trange(20):
    state=env.reset()
    done=False
    tot_reward=0
    for i in range(100):
        if not done:
            state=torch.tensor(state,dtype=torch.float32)
            q,action=select_action(state,episode)
            next_state,reward,done,info=env.step(env.actions[action.item()])
            tot_reward+=reward
            with torch.no_grad():
                next_q=neural_network(torch.tensor(next_state,dtype=torch.float32))
                max_next_q=torch.max(next_q)
            target=reward+GAMMA*max_next_q
            loss=loss_function(q,target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            state=next_state
    mean_rewards.append(tot_reward/(i+1))

plt.plot(mean_rewards)
plt.show()
plt.close()

plt.plot(losses)
plt.show()
plt.close()
