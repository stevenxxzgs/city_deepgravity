
import shap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

area_names = ['xingyang', 'jinshui', 'erqi', 'dengfeng', 'xinzheng', 'zhongmou', 'zhongyuan'] 
for area_name in area_names:
    # 读取数据
    test_path = f'../input_data/{area_name}/{area_name}_od.csv'
    df = pd.read_csv(test_path)
    df = df.dropna()

    # 定义输入和输出特征
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

    # 提取特征和目标值
    X_train = df[input_features].values
    y = df[output_feature].values

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.05, random_state=32)

    # 转换为 PyTorch 的 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 定义神经网络模型
    class FlowPredictionNN(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(FlowPredictionNN, self).__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, hidden_dim)
            self.layer4 = nn.Linear(hidden_dim, hidden_dim)
            self.layer5 = nn.Linear(hidden_dim, hidden_dim)
            self.layer6 = nn.Linear(hidden_dim, hidden_dim//2)
            self.layer7 = nn.Linear(hidden_dim//2, hidden_dim//2)
            self.layer8 = nn.Linear(hidden_dim//2, hidden_dim//2)
            self.layer9 = nn.Linear(hidden_dim//2, hidden_dim//2)
            self.layer10 = nn.Linear(hidden_dim//2, hidden_dim//2)
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

    # 定义模型
    input_dim = len(input_features)
    hidden_dim = 1024
    model = FlowPredictionNN(input_dim, hidden_dim)
    
    model.load_state_dict(torch.load(f'../output_data/{area_name}/{area_name}_flow_generation_model_nopre.pth'))
    model.eval()

    def model_fn(x):
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(x).numpy()


    # 创建解释器
    explainer = shap.KernelExplainer(model_fn, X_test_tensor.numpy()[:5])

    # 计算SHAP值
    shap_values = explainer.shap_values(X_test_tensor.numpy())
    print(shap_values)
    print(shap_values.shape)
    shap_values = np.squeeze(shap_values, axis=-1)

    # 创建 Explanation 对象
    shap_values_expl = shap.Explanation(
        values=shap_values,  # 选择第一个维度
        base_values=explainer.expected_value,
        data=X_test_tensor.numpy(),
        feature_names=input_features
    )
    print(shap_values_expl)
    print(shap_values_expl.shape)

    try:
        shap.plots.beeswarm(shap_values_expl)
    except ImportError as e:
        print(e)
        import importlib
        importlib.reload(shap)
        shap.plots.beeswarm(shap_values_expl, area_name=area_name)