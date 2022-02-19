#nn_configuration.py

import torch, torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
        nn.Conv2d( 1, 8, 5 ),
        nn.ReLU(),
        nn.MaxPool2d( 2, 2 ),

        nn.Conv2d( 8, 32, 5 ),
        nn.ReLU(),
        nn.MaxPool2d( 2, 2 ),
        
        nn.Flatten( start_dim = 1 ),

        nn.Linear( 32 * 29 * 29, 1024 ),
        nn.ReLU(),

        nn.Linear( 1024, 256 ),
        nn.ReLU(),

        nn.Linear( 256, 2 ),

        )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam( model.parameters(), lr = 0.001, weight_decay = 0.0001 )
