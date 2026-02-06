

import torch
import torch.nn as nn
import torch.optim as optim


class AcademicRepresentationNetwork(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.batch_norm = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batch_norm(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class NeuralTrainingLoop:

    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-3
        )

    def train(self, dataloader, epochs: int = 40):
        self.model.train()

        for epoch in range(epochs):
            cumulative_loss = 0.0

            for batch_idx, (X, y) in enumerate(dataloader):
                self.optimizer.zero_grad()

                logits = self.model(X)
                loss = self.loss_fn(logits.squeeze(), y.float())

                loss.backward()
                self.optimizer.step()

                cumulative_loss += loss.item()

            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"Loss: {cumulative_loss:.6f}"
            )
