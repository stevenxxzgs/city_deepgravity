import pandas as pd

# 指定CSV文件路径
csv_file_path = '/Users/steven/Code/mobility/DeepGravity/input_data/Villages_lat_lon_with_tile_ID.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 定义给定的经纬度
reference_longitude = 112.992031
reference_latitude = 34.714904

# 筛选出左上方的点
filtered_df = df[ (df['latitude'] >= reference_latitude)]

# 显示筛选后的DataFrame
print(filtered_df)

# 可选：将筛选后的数据保存为一个新的CSV文件
filtered_df.to_csv('../input_data/filtered_points.csv', index=False)