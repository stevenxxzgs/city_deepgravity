import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def common_part_of_commuters(values1, values2, numerator_only=False):
    if numerator_only:
        tot = 1.0
    else:
        tot = (np.sum(values1) + np.sum(values2))
    if tot > 0:
        return 2.0 * np.sum(np.minimum(values1, values2)) / tot
    else:
        return 0.0

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path).dropna()
    df_test = pd.read_csv(test_path).dropna()
    return df_train, df_test

def preprocess_data(df_train, df_test, input_features, output_feature):
    X_train = df_train[input_features].values
    y_train = df_train[output_feature].values
    X_test = df_test[input_features].values
    y_test = df_test[output_feature].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler

def create_model(input_dim, hidden_dim):
    class FlowPredictionNN(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(FlowPredictionNN, self).__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, hidden_dim)
            self.layer4 = nn.Linear(hidden_dim, hidden_dim)
            self.layer5 = nn.Linear(hidden_dim, hidden_dim)
            self.layer6 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.layer7 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.layer8 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.layer9 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.layer10 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.layer11 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.layer12 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.layer13 = nn.Linear(hidden_dim // 2, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.relu(self.layer3(x))
            x = self.dropout(x)
            x = self.relu(self.layer4(x))
            x = self.dropout(x)
            x = self.relu(self.layer5(x))
            x = self.dropout(x)
            x = self.relu(self.layer6(x))
            x = self.dropout(x)
            x = self.relu(self.layer7(x))
            x = self.dropout(x)
            x = self.relu(self.layer8(x))
            x = self.dropout(x)
            x = self.relu(self.layer9(x))
            x = self.dropout(x)
            x = self.relu(self.layer10(x))
            x = self.dropout(x)
            x = self.relu(self.layer11(x))
            x = self.dropout(x)
            x = self.relu(self.layer12(x))
            x = self.dropout(x)
            x = self.layer13(x)
            return x
    
    return FlowPredictionNN(input_dim, hidden_dim)

def train_model(model, X_train_tensor, y_train_tensor, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    return loss_history

def plot_loss(loss_history, loss_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path, dpi=300) 

def evaluate_model(model, X_test_tensor, y_test_tensor, scaler, area_name):
    model_path = f'../output_data/{area_name}/{area_name}_flow_generation_model_delsome.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_flows = predictions.numpy().flatten()
        test_loss = criterion(predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    actual_flows = y_test_tensor.numpy().flatten()
    cpc_test = common_part_of_commuters(actual_flows, predicted_flows, numerator_only=False)
    print(f'Test CPC: {cpc_test:.4f}')
    
    return predicted_flows

def save_results(df_test, predicted_flows, output_path):
    df_test['predicted_flows'] = pd.Series(predicted_flows)
    df_test.to_csv(output_path, index=False)

if __name__ == "__main__":
    # input_features = [
    #     "distance", "area_o", "road_o", "arch_o", "ndvi_o",
    #     "nightlight_o", "price_o", "altitude_o", "slope_o", "rain_o", "river_o",
    #     "hospital_o", "archhigh_o", "avg_gdp_o", "park_o", "shop_o", "company_o",
    #     "hotel_o", "bus_o", "archdensity_o", "population_o", "area_d", "road_d",
    #     "arch_d", "ndvi_d", "nightlight_d", "price_d", "altitude_d", "slope_d",
    #     "rain_d", "river_d", "hospital_d", "archhigh_d", "avg_gdp_d", "park_d",
    #     "shop_d", "company_d", "hotel_d", "bus_d", "archdensity_d", "population_d"
    # ]
    input_features = [
        "road_o", "arch_o", "population_o", "road_d", "arch_d", "population_d"
    ]
    output_feature = ["pop_flow_aft"]
    
    area_names = ['xingyang', 'jinshui', 'erqi', 'dengfeng', 'xinzheng', 'zhongmou', 'zhongyuan'] 
    for area_name in area_names:
        train_path = f'../input_data/{area_name}/without_{area_name}_od.csv'
        test_path = f'../input_data/{area_name}/{area_name}_od.csv'
        output_path = f'../output_data/{area_name}/{area_name}_result.csv'
        
        df_train, df_test = load_data(train_path, test_path)
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler = preprocess_data(df_train, df_test, input_features, output_feature)
        
        input_dim = len(input_features)
        hidden_dim = 1024
        model = create_model(input_dim, hidden_dim)
        
        loss_history = train_model(model, X_train_tensor, y_train_tensor)
        ensure_directory_exists(f'../output_data/{area_name}')
        plot_loss(loss_history, loss_path=f'../output_data/{area_name}/{area_name}_add_pre_loss_del.png')
        
        torch.save(model.state_dict(), f'../output_data/{area_name}/{area_name}_flow_generation_model_delsome.pth')
        
        predicted_flows = evaluate_model(model, X_test_tensor, y_test_tensor, scaler, area_name)
        save_results(df_test, predicted_flows, output_path)