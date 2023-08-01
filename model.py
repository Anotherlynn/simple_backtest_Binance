import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F

from sklearn.model_selection import cross_val_score

import pandas as pd
import os
from _func.Y import *
from alpha.alpha_MACD import *
from alpha.alpha_SWING import *
# import talib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################
# test sample data
folder_path = './data'
year = '2023'
file_name = 'binance_ADA_USDT_2018_2023.csv'
data = pd.read_csv(os.path.join(folder_path, year, file_name),index_col=0)
train = data[data.Timestamp <= '2023-01-20']
test = data[data.Timestamp > '2023-01-20']
train = train.drop("Timestamp", axis=1)
test = test.drop("Timestamp", axis=1)

# Generate target for different model type in the first place
def y_generator(input, type, mark_out):
    """
    Define target for different models
    :param input: input, type
    :param type: pd.Dataframe, string
    :return: X, y(target)
    """
    if type == 'Regression' or 'reg':
        # diff ratio of price from previous bar
        # defined according to https://www.tradingview.com/script/l6A4RBBo-Difference-over-bars/
        target = DIFF(input).diff(mark_out, use_ma=False)
    elif type == 'Classification' or 'cf':
        # chance of going up or down chance of going up or down
        # define according to https://www.tradingview.com/script/1FLTUW2e-Bar-Move-Probability-BMP/
        target = BMP(input, length=800, mark_out=mark_out).cal_dist()

    return target


class MyModels:
    def __init__(self, task, input_size, loss=None):
        self.task = task # regression or classfication
        self.input_size = input_size
        # should be X.shape[2]
        self.loss = loss # self defined loss

    def lossform(self):
        loss_ =  nn.MSELoss() \
            if self.task == 'reg' \
            else (nn.CrossEntropyLoss()
                  if self.task == 'cf'
                  else self.loss)
        return loss_

    def lasso(self,alpha):
        lasso = nn.Linear(*self.input_size,1,bias=True)

        def forward(x):
            x = self.lasso(x)
            # L1 = alpha * torch.norm(lasso.weight, p=1)
            return x

        return lasso

    def decision_tree(self, hidden_size, num_classes):
        num = num_classes if self.task == 'cl' else 1
        linear1 = nn.Linear(self.input_size, hidden_size)
        relu = nn.ReLU()
        linear2 = nn.Linear(hidden_size, num)

        def forward(x):
            x = linear1(x)
            x = relu(x)
            x = linear2(x)
            return x

    def randomForest(self, num_trees, hidden_size, num = 0):
        trees = [self.decision_tree(self.input_size, hidden_size, num) for _ in range(num_trees)]

        def forward(x):
            predictions = torch.stack([tree(x) for tree in trees], dim=2)
            if self.task == 'reg':
                return torch.mean(predictions, dim=2)
            else:
                return torch.mode(predictions, dim=2).values
        return forward

    def LSTM(self, hidden_size =64, output_size=1):
        lstm = nn.LSTM(self.input_size, hidden_size, batch_first=True)
        fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            _, (h_n, _) = lstm(x)
            out = fc(h_n[-1])
            return out
        return forward

    def GRU(self, hidden_size = 64, output_size=1):
        gru = nn.GRU(self.input_size, hidden_size, batch_first=True)
        fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            _, (h_n, _) = gru(x)
            out = fc(h_n[-1])
            return out

        return forward

    def CNN(self, kernel=3, hidden_size = 64, output_size=1):
        # Define the layers of the CNN
        conv1 = nn.Conv2d(self.input_size, hidden_size, kernel_size=kernel, stride=1, padding=1)
        relu1 = nn.ReLU()
        maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        relu2 = nn.ReLU()
        maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        flatten = nn.Flatten()
        fc1 = nn.Linear(32 * 7 * 7, hidden_size)
        relu3 = nn.ReLU()
        fc2 = nn.Linear(hidden_size, output_size)

        def forward(x):
            x = conv1(x)
            x = relu1(x)
            x = maxpool1(x)
            x = conv2(x)
            x = relu2(x)
            x = maxpool2(x)
            x = flatten(x)
            x = fc1(x)
            x = relu3(x)
            x = fc2(x)
            return x

        return forward

    def transformer_regression(self, num_heads=8, num_layers=6, num_classes=3, hidden_dim=2048):
        # Define the Transformer layers
        num = num_classes if self.task == 'cl' else 1
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=num_heads, dim_feedforward=hidden_dim)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        linear = nn.Linear(self.input_size, num)

        def forward(x):
            x = transformer_encoder(x)
            x = torch.mean(x, dim=1)
            x = linear(x)
            if self.task == 'cl':
                x = F.softmax(x, dim=1)
                return x
            else: return x.squeeze()
        return forward()


# Load and preprocess your data
X_train = train  # Input features
y_regression = y_generator(X_train, 'Regression', 5)  # Regression target variable (difference ratio)
y_classification = y_generator(X_train, 'Classification', 5)  # Classification target variable (up/down chance)

# Convert data to PyTorch tensors
X = torch.tensor(X_train.to_numpy().astype(np.float32), dtype=torch.float32).to(device)
y_regression = torch.tensor(y_regression, dtype=torch.float32).to(device).unsqueeze(1)
y_classification = torch.tensor(y_classification, dtype=torch.long).to(device).unsqueeze(1)

# Train the model on GPU
learning_rate = 2e-5
Model = MyModels(task = 'reg', input_size = (X.shape[1],))
# choose and change your model here
model = Model.lasso(0.01).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = Model.lossform()

epochs = 10


for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    if Model.task == 'reg':
        loss = criterion(outputs, y_regression)
    else:
        loss = criterion(outputs.squeeze(), y_classification)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

predictions = model(X).detach().numpy().flatten()


# Cross-Validation
cv_scores = cross_val_score(y_regression, X.numpy(), y_regression.numpy(), cv=5, scoring='neg_mean_squared_error')
mean_mse = np.mean(-cv_scores)  # Mean squared error (negative due to scoring convention)

# Training with different markouts
markouts = [5, 10, 20, 40, 60, 80, 100]
for markout in markouts:
    X_markout = ...  # Preprocess data with the specific markout
    # Convert data to PyTorch tensors
    X_markout = torch.tensor(X_markout, dtype=torch.float32)
    y_regression_markout = ...  # Regression target variable with the specific markout
    y_regression_markout = torch.tensor(y_regression_markout, dtype=torch.float32)
    # Train models and evaluate performance
    # ...

