import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

MODEL_DIR_PATH = "./model"


class Linear_QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        if not os.path.exists(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)

        file_name = os.path.join(MODEL_DIR_PATH, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="model.pth"):
        if os.path.exists(os.path.join(MODEL_DIR_PATH, file_name)):
            file_name = os.path.join(MODEL_DIR_PATH, file_name)
            return torch.load(file_name)


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.model: Linear_QNet = model
        self.lr: float = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        # 1: Predicted Q values with current state
        pred: torch.Tensor = self.model(state)

        # 2: Q_new = r + y * max(next_predicted Q values) -> Only do this if not done
        # pred.clone()
        # predictions[argmax(action)] = Q_new
        target = pred.clone()
        for i, _ in enumerate(done):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss: torch.Tensor = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
