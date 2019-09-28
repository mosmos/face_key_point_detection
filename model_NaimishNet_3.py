

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


    


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        self.conv1 = nn.Conv2d(1,   32,  7)   # out> 32  *218 *218 #pool> 32  *109 *109
        self.conv2 = nn.Conv2d(32,  64,  3)   # out> 64  *107 *107 #pool> 64  *53  *53
        self.conv3 = nn.Conv2d(64,  128, 2)   # out> 128 *52  *52  #pool> 128 *26  *26
        self.conv4 = nn.Conv2d(128, 256, 1)   # out> 256 *26  *23  #pool> 256 *13  *13

        self.pool = nn.MaxPool2d(2,2) 
        
        self.conv_drop = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(256*13*13, 256*13)
        self.fc1_drop = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(256*13, 256*6)
        self.fc2_drop = nn.Dropout(p=0.4)
        
        self.fc3 = nn.Linear(256*6, 68*2)
        

        
    def forward(self, x):

        x = self.pool(F.elu(self.conv1(x)))
        x = self.conv_drop(self.pool(F.elu(self.conv2(x))))
        x = self.conv_drop(self.pool(F.elu(self.conv3(x))))
        x = self.conv_drop(self.pool(F.elu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        
        x = self.fc1_drop(F.selu(self.fc1(x)))
        x = self.fc2_drop(F.selu(self.fc2(x)))

        x = self.fc3(x)
        
        return x