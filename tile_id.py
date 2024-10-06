import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 读取泰森多边形的Shapefile文件
tessellation_path = "./tessellation.shp"
tessellation = gpd.read_file(tessellation_path)


# 读取CSV文件
data = pd.read_csv('../input_data/Villages_lat_lon.csv', header=None, names=['village_name', 'latitude', 'longitude', 'ID'])

# 打印DataFrame的前几行
print(data.head())

# 打印每一列的数据类型
print(data.dtypes)

# 检查并转换latitude和longitude列
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        print(f"无法转换值: {value}")
        return None

data['latitude'] = data['latitude'].apply(convert_to_float)
data['longitude'] = data['longitude'].apply(convert_to_float)

# 删除转换失败的行
data = data.dropna(subset=['latitude', 'longitude'])

# 打印转换后的DataFrame的前几行
print(data.head())

# 创建Point几何对象
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]

# 创建GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# 提取点的坐标
points = gdf[['longitude', 'latitude']].values


# 查找每个村庄所属的泰森多边形
def find_tile_id(point, tessellation):
    for idx, row in tessellation.iterrows():
        if row['geometry'].contains(point):
            return row['tile_ID']
    return None

# 应用查找函数
gdf['tile_ID'] = gdf.apply(lambda row: find_tile_id(row['geometry'], tessellation), axis=1)

# 保存结果到新的CSV文件
output_path = '../input_data/Villages_lat_lon_with_tile_ID.csv'
gdf.to_csv(output_path, index=False)

print(f"新的CSV文件已保存至: {output_path}")