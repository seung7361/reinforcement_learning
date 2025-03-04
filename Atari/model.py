import torch
import random


class QNetwork(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # in_dim: (B, 4, 84, 84)
        # out_dim: (action_space.n, )
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim[0], 32, 8, 4), # (32, 20, 20)
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2), # (64, 9, 9)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1), # (64, 7, 7)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_dim)
        )
    
    def forward(self, x):
        # x: (B, 4, 84, 84)
        x = self.conv(x)

        x = x.view(x.shape[0], -1)
        
        x = self.dense(x)
        # x: (B, out_dim)
        return x

    def action(self, state, epsilon, device):
        if random.random() > epsilon:
            q_value = self(state) # (1, 4) -- argmax --> (1, 1)
            action = q_value.argmax(dim=-1).item()
        else:
            action = random.randrange(0, self.out_dim)


        return action