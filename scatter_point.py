import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import DBSCAN
from shapely.geometry import Point

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

# 定义eps和min_samples的组合
eps_values = [0.02]
min_samples_values = [5]

# 遍历所有组合并绘制散点图
for eps in eps_values:
    for min_samples in min_samples_values:
        # 进行DBSCAN聚类
        X = gdf_data[['lat_o', 'lon_o']].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_

        # 将聚类结果添加到GeoDataFrame
        gdf_data['cluster'] = labels

        # 绘制地图
        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制Shapefile
        gdf.plot(ax=ax, color='white', edgecolor='black', alpha=0.18)

        # 绘制散点图，不同聚类用不同颜色表示
        for cluster in set(labels):
            if cluster == -1:
                color = 'gray'  # 噪声点
                gdf_data[gdf_data['cluster'] == cluster].plot(ax=ax, markersize=gdf_data['size'], alpha=0.2, color=color, edgecolor='black')

            else:
                color = plt.cm.get_cmap('tab20')(cluster % 20)
                gdf_data[gdf_data['cluster'] == cluster].plot(ax=ax, markersize=gdf_data['size'], alpha=0.9, color=color, edgecolor='black')


        # 添加标题和标签
        ax.set_title(f"Predicting outflow point with DBSCAN Clusters (eps={0.018}, min_samples={min_samples})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # 保存图片
        plt.savefig(f"./scatter/scatter_plot_flow_aggregated_with_shp_dbscan_eps_{eps}_min_samples_{min_samples}.png", dpi=300, bbox_inches='tight')

        # 显示图片
        # plt.show()