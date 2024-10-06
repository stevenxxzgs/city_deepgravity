import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

def common_part_of_commuters(values1, values2, numerator_only=False):
    if numerator_only:
        tot = 1.0
    else:
        tot = (np.sum(values1) + np.sum(values2))
    if tot > 0:
        return 2.0 * np.sum(np.minimum(values1, values2)) / tot
    else:
        return 0.0

df_train = pd.read_csv('../input_data/select_train.csv').dropna()
df_test = pd.read_csv('../input_data/select_test.csv').dropna()

#
input_features = [
    "distance", "area_o", "road_o", "arch_o", "ndvi_o",
    "nightlight_o", "price_o", "altitude_o", "slope_o", "rain_o", "river_o",
    "hospital_o", "archhigh_o", "avg_gdp_o", "park_o", "shop_o", "company_o",
    "hotel_o", "bus_o", "archdensity_o", "population_o", "area_d", "road_d",
    "arch_d", "ndvi_d", "nightlight_d", "price_d", "altitude_d", "slope_d",
    "rain_d", "river_d", "hospital_d", "archhigh_d", "avg_gdp_d", "park_d",
    "shop_d", "company_d", "hotel_d", "bus_d", "archdensity_d", "population_d"
]

output_feature = ["pop_flow_aft"]

# 获取特征和标签
X_train = df_train[input_features].values
y_train = df_train[output_feature].values
X_test = df_test[input_features].values
y_test = df_test[output_feature].values

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# 转换为 PyTorch 的 Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class FlowPredictionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FlowPredictionNN, self).__init__()
        # self.layer1 = nn.Linear(input_dim, hidden_dim)
        # self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer6 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer7 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer8 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer9 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer10 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer11 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer12 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer13 = nn.Linear(hidden_dim, 1)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer6 = nn.Linear(hidden_dim, hidden_dim//2)
        self.layer7 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.layer8 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.layer9 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.layer10= nn.Linear(hidden_dim//2, hidden_dim//2)
        self.layer11 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.layer12 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.layer13 = nn.Linear(hidden_dim//2, 1)
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


input_dim = len(input_features)
hidden_dim = 1024
model = FlowPredictionNN(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
loss_history = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), '../output_data/flow_generation_model.pth')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# # 载入模型
# model.load_state_dict(torch.load('flow_prediction_model.pth'))
# model.eval()

# # 对测试集进行预测
# with torch.no_grad():
#     predictions = model(X_test_tensor)
#     test_loss = criterion(predictions, y_test_tensor)
#     print(f'Test Loss: {test_loss.item():.4f}')

# # 生成新的国家城市间流量
# def generate_flow(city_data):
#     city_data_scaled = scaler.transform(city_data)
#     city_data_tensor = torch.tensor(city_data_scaled, dtype=torch.float32)
#     with torch.no_grad():
#         flow_predictions = model(city_data_tensor)
#     return flow_predictions.numpy()

# new_city_data = pd.read_csv('/Users/steven/Code/mobility/DeepGravity/data_week2/ZhengZhou/test_set.csv')
# new_city_data_features = new_city_data[input_features].values
# generated_flows = generate_flow(new_city_data_features)

# # 保存生成的流量
# new_city_data['predicted_flow'] = generated_flows
# new_city_data.to_csv('predicted_flows.csv', index=False)


##################

model.load_state_dict(torch.load('../output_data/flow_generation_model.pth'))
model.eval()

with torch.no_grad():
    y_torch = y_test_tensor
    X_test_scal = X_test_scaled.astype(np.float32)
    X_torch = torch.tensor(X_test_scal, dtype=torch.float32)

    predictions = model(X_torch)
    print(predictions.size())
    predicted_flows = predictions.numpy()
    flows_list = predicted_flows.tolist()
    with open('../output_data/predicted.csv', 'w') as file:
        for flow in flows_list:
            file.write(f"{flow}\n")


    test_loss = criterion(predictions, y_torch)
    print(f'Test Loss: {test_loss.item():.4f}')


actual_flows = y_test
cpc_test= common_part_of_commuters(actual_flows, predicted_flows, numerator_only=False)
print(f'Test CPC: {cpc_test}')

df_test['predicted_flows'] = pd.Series(flows_list)
df_test.to_csv('../output_data/test_result.csv', index=False)

# def generate_flow(city_data):
#     city_data_scaled = scaler.transform(city_data)
#     city_data_tensor = torch.tensor(city_data_scaled, dtype=torch.float32)
#     with torch.no_grad():
#         flow_predictions = model(city_data_tensor)
#     return flow_predictions.numpy()

# new_city_data = pd.read_csv('/Users/steven/Code/mobility/DeepGravity/data_week2/California/test_set.csv')
# new_city_data_features = new_city_data[input_features].values
# generated_flows = generate_flow(new_city_data_features)
# new_city_data['predicted_flow'] = generated_flows
# new_city_data.to_csv('predicted_flows.csv', index=False)
# actual_flows_new = new_city_data['total_pop_flow'].values 

# cpc_new= common_part_of_commuters(actual_flows_new, generated_flows, numerator_only=False)
# print(f'New Data CPC: {cpc_new}')