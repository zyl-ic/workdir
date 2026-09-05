import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    """MAPPO 的中心化 value 网络：输入全局 state，输出标量 V(s)。

    CTDE：训练时 critic 用全局 state（centralized），执行时 actor 只用局部 obs（decentralized）。
    """

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        input_shape = scheme["state"]["vshape"]  # state_size
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, states):
        # states: (..., state_size) -> (..., 1)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
