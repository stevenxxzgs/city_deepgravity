# import pandas as pd
# import folium
# from folium.plugins import MarkerCluster

# # 读取Excel文件
# file_path = '/Users/steven/Code/mobility/DeepGravity/input_data/origin.xlsx'
# df = pd.read_excel(file_path)

# # 获取中心点
# center_lat = df['_y'].mean()
# center_lon = df['_x'].mean()

# # 创建地图
# m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# # 添加MarkerCluster插件以聚合标记点
# marker_cluster = MarkerCluster().add_to(m)

# # 在地图上绘制所有点
# for index, row in df.iterrows():
#     lat, lon = row['_y'], row['_x']
#     folium.Marker(
#         location=[lat, lon],
#         popup=f"Latitude: {lat}, Longitude: {lon}",
#         icon=folium.Icon(color='blue')
#     ).add_to(marker_cluster)

# # 保存地图为HTML文件
# m.save("../output_data/output_map.html")

# 如果需要保存为图片，可以使用screenshot工具或其他方法
# 这里是一个简单的命令行截图工具的示例
# 注意：这需要安装selenium和PIL库，并且需要一个浏览器驱动程序（如ChromeDriver）
# pip install selenium pillow
# 从selenium获取截图


# 金水区
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 读取Excel文件
file_path = '/Users/steven/Code/mobility/DeepGravity/input_data/origin.xlsx'
df = pd.read_excel(file_path)

# 确定金水区的边界范围
# 假设金水区的边界范围如下：
jinshui_min_lat = 34.733   # 最南端纬度
jinshui_max_lat = 34.867   # 最北端纬度
jinshui_min_lon = 113.600  # 最西端经度
jinshui_max_lon = 113.840  # 最东端经度

# 筛选出金水区的数据点
df_jinshui = df[(df['latitude'] >= jinshui_min_lat) & (df['latitude'] <= jinshui_max_lat) &
                (df['longitude'] >= jinshui_min_lon) & (df['longitude'] <= jinshui_max_lon)]
df_jinshui_output = df_jinshui['ID2']
df_jinshui_output.to_csv('../output_data/jinshui.csv', index=False)
# 获取中心点
center_lat = df_jinshui['latitude'].mean()
center_lon = df_jinshui['longitude'].mean()

# 创建地图
m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# 添加MarkerCluster插件以聚合标记点
marker_cluster = MarkerCluster().add_to(m)

# 在地图上绘制所有点
for index, row in df_jinshui.iterrows():
    lat, lon = row['latitude'], row['_x']
    folium.Marker(
        location=[lat, lon],
        popup=f"Latitude: {lat}, Longitude: {lon}",
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

# 保存地图为HTML文件
m.save("../output_data/output_map.html")