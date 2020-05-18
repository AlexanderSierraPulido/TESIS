import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

file_name = "waveforms.hdf5"
csv_file = "metadata.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')
# filterering the dataframe
df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
print(f'total events selected: {len(df)}')

# making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()

# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')
DataArray = np.zeros((len(df),3,6000))
arrival = np.zeros((len(df),2))

for i, evi in enumerate(ev_list):
    dataset = dtfl.get('earthquake/local/'+str(evi))
    # waveforms, 3 channels: first row: E channle, second row: N channel, third row: Z channel
    data = np.array(dataset)
            
    DataArray[i,0,:] = data[:,0]
    DataArray[i,1,:] = data[:,1]
    DataArray[i,2,:] = data[:,2]
    arrival[i,0] = dataset.attrs['p_arrival_sample']
    arrival[i,1] = dataset.attrs['s_arrival_sample']


Tensor = torch.tensor(DataArray)
Tensor = torch.unsqueeze(Tensor,1)

data_picking = torch.tensor(arrival)


class NN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (2,501))
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size = (2,493))

        self.fc1 = nn.Linear(in_features = 2*2*4992, out_features = 3)
        self.fc1 = nn.Linear(in_features = 3, out_features = 5)
        self.out = nn.Linear(in_features = 5, out_features = 6000)

    def forward(self, tensor):
        t = tensor
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = (2,10), stride = (10,1), padding = (10,1))
    
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = (1,10), stride = (10,1), padding = (10,1))

        t = reshape(-1, 50*2*4992)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        t = F.softmax(t)    
        return t

train1 = NN_Network()
loss = 0
correct = 0

optimizer = optim.Adam(train1.parameters(),lr=0.01)

prediction = train1(Tensor)  # here is were data comes into the network
loss = F.cross_entropy(prediction,data_picking)

optimizer.zero_grad()
loss.backward() # Calculate gradients
optimizer.step() #Update learneable parameters

loss += loss.item()
correct += prediction.argmax(dim = 1).eq(data_picking).sum().item()
print (correct, loss)
