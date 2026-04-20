import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np


# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x


def preprocess(df):

    df.rename(columns={
        'temperature_2m (°C)': 'temperature',
        'cloudcover_low (%)': 'cloudcover_low',
        'cloudcover_mid (%)': 'cloudcover_mid',
        'cloudcover_high (%)': 'cloudcover_high',
        'windspeed_10m (km/h)': 'windspeed', 
        'winddirection_10m (°)': 'winddirection',
    }, inplace=True)

    # map hvfhs_license_num to service type
    service_mapping = {
        'HV0002': 'juno',
        'HV0003': 'uber',
        'HV0004': 'via',
        'HV0005': 'lyft'
    }

    df['service_name'] = df['hvfhs_license_num'].map(service_mapping)

    one_hot_encoding = pd.get_dummies(df['service_name'], prefix='is')
    df = pd.concat([df, one_hot_encoding], axis=1)

    df = df.loc[df['base_passenger_fare'] > 0]
    df['tip_percent'] = df['tips'] / (df['base_passenger_fare'] + df['tolls'] + df['sales_tax'])

    df['shared_with_friend'] = (df['shared_request_flag'] == "Y") & (df['shared_match_flag'] == "N")
    df['shared_with_stranger'] = df['shared_match_flag'] == "Y"

    # drop non-numeric and redundant columns
    df.drop(columns={
        'hvfhs_license_num', 
        'dispatching_base_num', 
        'originating_base_num',
        'request_datetime', 
        'on_scene_datetime', 
        'pickup_datetime',
        'dropoff_datetime',
        'shared_request_flag',
        'shared_match_flag',
        'access_a_ride_flag',
        'wav_request_flag',
        'wav_match_flag',
        'time',
        'service_name',
        'airport_fee'
    }, inplace=True)

    print("preprocessing done....")

    return df


def train(df):
    # hyperparameters
    batch_size = 128
    lr = 0.001

    # separate features from output class label
    x = df.drop(columns=['tip_percent']).values.astype(np.float32)
    y = df['tip_percent'].values.astype(np.float32)

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNet(x_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    epochs = 100
    try:
        for epoch in tqdm(range(epochs)):
            model.train()

            epoch_loss = 0
            for batch_x, batch_y in train_loader:

                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate batch loss
                epoch_loss += loss.item()
            
            # average loss for the epoch
            if (epoch+1) % 10 == 0:
                print(f'\nEpoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}')
    except KeyboardInterrupt:
        # allows user to stop training early, while still saving and testing the model
        pass
    
    torch.save(model, os.path.join(__file__, '../model.pth'))
    print("model saved.")

    test(model, x_test, y_test)



def test(model, x_test, y_test):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs.squeeze(), y_test)
        print(f'Test Loss: {test_loss.item()}')


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(__file__, "../sampled_taxi_weather_data.csv"))
    df = preprocess(df)

    train(df)
