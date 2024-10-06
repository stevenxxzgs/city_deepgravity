import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from scipy.spatial import Voronoi
import numpy as np

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

# 生成Voronoi图
vor = Voronoi(points)

# 定义一个函数来创建Voronoi多边形，并处理无限区域
def create_voronoi_polygons(vor, boundary):
    polygons = []
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not -1 in region and len(region) > 0:
            polygon = Polygon([vor.vertices[i] for i in region])
            # 裁剪多边形以适应边界
            try:
                clipped_polygon = polygon.intersection(boundary).buffer(0)
                if not clipped_polygon.is_empty:
                    polygons.append(clipped_polygon)
                else:
                    print(f"Clipped polygon is empty for point index {point_idx}")
            except Exception as e:
                print(f"Error clipping polygon for point index {point_idx}: {e}")
        else:
            # 对于包含-1的区域，尝试生成多边形
            new_polygon = create_infinite_voronoi_polygon(vor, region, boundary, point_idx)
            if new_polygon:
                try:
                    polygons.append(new_polygon.buffer(0))
                except Exception as e:
                    print(f"Error buffering polygon for point index {point_idx}: {e}")
    return polygons

# 处理无限区域的函数
def create_infinite_voronoi_polygon(vor, region, boundary, point_idx):
    if -1 in region:
        # 获取有限顶点
        finite_vertices = [vor.vertices[i] for i in region if i != -1]
        # 获取无限顶点的方向向量
        infinite_edges = [i for i, edge in enumerate(vor.ridge_vertices) if -1 in edge and set(edge) & set(region)]
        for edge_idx in infinite_edges:
            start, end = vor.ridge_vertices[edge_idx]
            if start == -1:
                start, end = end, start
            # 计算方向向量
            direction = vor.vertices[end] - vor.vertices[start]
            # 扩展方向向量以确保多边形覆盖边界
            extended_vertex = vor.vertices[start] + 10 * direction
            finite_vertices.append(extended_vertex)
        # 创建多边形
        polygon = Polygon(finite_vertices)
        # 裁剪多边形以适应边界
        try:
            clipped_polygon = polygon.intersection(boundary).buffer(0)
            if not clipped_polygon.is_empty:
                return clipped_polygon
        except Exception as e:
            print(f"Error creating infinite polygon for point index {point_idx}: {e}")
    return None

# 定义一个边界框，用于裁剪无限的Voronoi区域
boundary = box(min(data['longitude']), min(data['latitude']), max(data['longitude']), max(data['latitude']))

# 创建Voronoi多边形
polygons = create_voronoi_polygons(vor, boundary)

# 确保polygons和data长度一致
if len(polygons) != len(data):
    print(f"Length of polygons ({len(polygons)}) does not match length of data ({len(data)}). Attempting to fix...")
    # 尝试手动处理缺失的多边形
    for i in range(len(data)):
        if i >= len(polygons):
            print(f"Missing polygon for point index {i}. Creating a fallback polygon.")
            point = Point(data.loc[i, 'longitude'], data.loc[i, 'latitude'])
            buffer_distance = 0.001  # 选择一个合适的缓冲距离
            fallback_polygon = point.buffer(buffer_distance)
            polygons.append(fallback_polygon)

# 将多边形添加到GeoDataFrame
gdf_voronoi = gpd.GeoDataFrame(data, geometry=polygons)
gdf_voronoi = gdf_voronoi.rename(columns={'ID': 'tile_ID'})

# 保存为shapefile
output_path = "./tessellation.shp"
gdf_voronoi.to_file(output_path, driver='ESRI Shapefile')

print(f"泰森多边形已保存至: {output_path}")