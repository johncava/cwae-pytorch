import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
from utility import *
import math
import numpy as np

dataset = create_dataset()

class VAE(nn.Module):

    def __init__(self):
        super(VAE,self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(784,784),
                nn.ReLU(),
                nn.Linear(784,500),
                nn.ReLU()
                )
        self.mu = nn.Linear(500,500)
        self.sigma = nn.Linear(500,500)
        self.decoder = nn.Sequential(
                nn.Linear(500,784),
                nn.ReLU(),
                nn.Linear(784,784),
                nn.ReLU()
        )

    def forward(self,x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        normal = Normal(mu,sigma)
        pred = normal.rsample()
        pred = self.decoder(pred)
        return pred, mu


model = VAE()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss_fn = nn.MSELoss()
epochs = 4
batch = 32

gamma = (4/(3*batch)) ** (2/5)
D = 784
total_loss = []
for epoch in range(epochs):
    epoch_loss = []
    for iteration in range(int(len(dataset)/batch)):
        x,ex = [],[]
        loss,loss1,loss2,loss3 = 0, 0, 0,0
        for item in dataset[iteration*batch: iteration*batch + batch]:
            x = Variable(torch.Tensor(item[1]), requires_grad = False)
            pred, mu = model(x)
            ex.append(mu)
            loss1 += loss_fn(pred,x)
        loss1 = torch.div(loss1,batch)
        for i in range(len(ex)):
            for j in range(len(ex)):
                x = ex[i]
                y = ex[j]
                loss2 += torch.rsqrt(torch.add(torch.div(torch.norm(x-y,2),2*D - 3),gamma))
        loss2 = torch.div(loss2, batch ** 2)
        for i in range(len(ex)):
            loss3 += torch.rsqrt(torch.add(torch.div(torch.norm(ex[i],2),2*D - 3),gamma + 0.5))
        loss3 = torch.mul(torch.div(loss3,batch),2)
        loss3 = torch.add(loss3, (1 + gamma) ** (-0.5))
        loss = loss1 + torch.log(torch.div(loss2 + loss3, 2 * math.sqrt(math.pi)))
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
        print(loss.item())
        loss.backward()
        optimizer.step()
        del x, ex
        #print('Iteration: ', iteration)
    total_loss.append(sum(epoch_loss)/len(epoch_loss))
    print('Epoch: ' + str(epoch), epoch_loss[iteration])
    del epoch_loss

print('Done')
