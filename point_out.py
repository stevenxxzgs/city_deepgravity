import pandas as pd
dfs = []
# 指定文件路径
area_names = ['xingyang', 'jinshui', 'erqi', 'xinmi', 'dengfeng', 'xinzheng', 'zhongmou', 'zhongyuan'] 
# area_name = 'your_area_name'  # 请替换为实际的区域名称
for area_name in area_names:
    test_records_path = f'../input_data/{area_name}/{area_name}_point.csv'
    df = pd.read_csv(test_records_path)
    dfs.append(df)
# 显示DataFrame
print(dfs)
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(f'../input_data/combined_points.csv', index=False)


import pandas as pd

point_file_path = f'/Users/steven/Code/mobility/DeepGravity/input_data/Villages_lat_lon.csv'
# point_file_path = f'/Users/steven/Code/mobility/DeepGravity/input_data/merged_od_filtered_points.csv'

# 获取combined_points.csv中的所有ID
valid_ids = combined_df['ID'].unique()

# 读取原始的点表文件
point_df = pd.read_csv(point_file_path)

# 筛选出包含这些ID的行
filtered_df = point_df[point_df['ID'].isin(valid_ids)]
# filtered_df = filtered_df[filtered_df['destination'.isin(valid_ids)]]
# 显示筛选后的DataFrame
print(filtered_df)

# 将筛选后的数据保存为一个新的CSV文件
filtered_df.to_csv(f'../input_data/Villages_lat_lon_filtered.csv', index=False)