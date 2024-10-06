import geopandas as gpd

# 指定Shapefile文件路径
shp_file_path = '/Users/steven/Code/mobility/DeepGravity/input_data/郑州市边界_410100_Shapefile_(poi86.com)/410100.shp'

# 读取Shapefile文件
gdf = gpd.read_file(shp_file_path)


# 提取索引为7的行
gdf_deleted_row = gdf.iloc[6:7]

# 将删除的行写入另一个Shapefile
gdf_deleted_row.to_file('gongyi.shp')


# 删除索引为7的行
gdf_new = gdf.drop(index=7)

# 重置索引（可选）
gdf_new = gdf_new.reset_index(drop=True)

# 将删除后的GeoDataFrame写入一个新的Shapefile
gdf_new.to_file('./without_gongyi.shp')

# 打印结果以确认
print("New GeoDataFrame written to new_file.shp")
print("Deleted row written to deleted_row.shp")