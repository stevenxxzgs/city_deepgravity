import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# 读取数据
data = pd.read_csv('/Users/steven/Code/mobility/DeepGravity/input_data/merged_od_data_612.csv')
data = data[['origin', 'lat_o', 'lon_o', 'flow']]

# 去重，确保每个origin对应的lat_o和lon_o是唯一的
data_unique = data.drop_duplicates(subset=['origin'])

# 聚合数据，将同一个origin的flow相加
aggregated_data = data.groupby('origin').agg({'flow': 'sum'}).reset_index()

# 将lat_o和lon_o合并到聚合后的数据中
aggregated_data = aggregated_data.merge(data_unique[['origin', 'lat_o', 'lon_o']], on='origin', how='left')

# 定义点的大小范围
min_size = 10
max_size = 11

# 计算点的大小
aggregated_data['size'] = ((aggregated_data['flow'] - aggregated_data['flow'].min()) / (aggregated_data['flow'].max() - aggregated_data['flow'].min())) * (max_size - min_size) + min_size

# 读取Shapefile
shapefile_path = 'merge.shp'
gdf = gpd.read_file(shapefile_path)

# 创建GeoDataFrame
gdf_data = gpd.GeoDataFrame(
    aggregated_data, 
    geometry=gpd.points_from_xy(aggregated_data.lon_o, aggregated_data.lat_o),
    crs=gdf.crs
)

# 绘制地图
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制Shapefile
gdf.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)

# 绘制散点图
gdf_data.plot(ax=ax, markersize=gdf_data['size'], alpha=0.8, color='red', edgecolor='black')

# 添加背景地图
# ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)

# 添加标题和标签
ax.set_title("Predicting outflow points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 保存图片
plt.savefig("./scatter_plot_flow_aggregated_with_shp.png", dpi=300, bbox_inches='tight')
