import geopandas as gpd
import pandas as pd

# 读取第一个 Shapefile 文件
gdf1 = gpd.read_file('zhengzhou_tessellation.shp')

# 读取第二个 Shapefile 文件
gdf2 = gpd.read_file('gongyi.shp')

# 统一 CRS
if gdf1.crs != gdf2.crs:
    gdf2 = gdf2.to_crs(gdf1.crs)

# 获取 gdf1 的所有列名
columns1 = gdf1.columns.tolist()
print(columns1, gdf2.columns)
# 在 gdf2 中添加缺失的列，并用默认值填充
for col in columns1:
    if col not in gdf2.columns:
        gdf2[col] = -1  # 或者用其他默认值，例如空字符串 '' 或 0

# 确保几何列名称一致
if gdf1.geometry.name != gdf2.geometry.name:
    gdf2 = gdf2.rename_geometry(gdf1.geometry.name)

# 重新排列 gdf2 的列顺序，使其与 gdf1 一致
gdf2 = gdf2[columns1]

# 合并两个 GeoDataFrame
merged_gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

# 保存合并后的 GeoDataFrame 到新的 Shapefile 文件
merged_gdf.to_file('merge.shp')

print("合并完成，已保存到 merge.shp")