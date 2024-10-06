import geopandas as gpd
from shapely.geometry import Polygon

# 指定泰森多边形文件的路径
tessellation_path = "./tessellation.shp"

# 读取泰森多边形数据
tessellation = gpd.read_file(tessellation_path).rename(columns={'tile_id': 'tile_ID'})

# 指定郑州市边界文件的路径
# boundary_path = "/Users/steven/Code/mobility/DeepGravity/input_data/郑州市边界_410100_Shapefile_(poi86.com)/410100.shp"
boundary_path = "./without_gongyi.shp"

# 读取郑州市边界数据
zhengzhou_boundary = gpd.read_file(boundary_path)


# 对泰森多边形和郑州市边界进行几何裁剪
zhengzhou_tessellation = gpd.overlay(tessellation, zhengzhou_boundary, how='intersection')

# 保存结果到新的文件
output_path = "./zhengzhou_tessellation.shp"
zhengzhou_tessellation.to_file(output_path, driver='ESRI Shapefile')

print(f"裁剪完成，郑州市的泰森多边形已保存至: {output_path}")