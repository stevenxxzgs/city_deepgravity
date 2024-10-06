import pandas as pd

# 读取第一个表格
od_df = pd.read_csv('../input_data/od_6-12.csv')

# 读取第二个表格
village_df = pd.read_csv('../input_data/Villages_lat_lon.csv')

# 重命名 village_df 的列，以便合并
village_df.rename(columns={'ID': 'O_code'}, inplace=True)

# 合并 O_code 对应的经纬度
od_df = od_df.merge(village_df[['O_code', 'latitude', 'longitude']], on='O_code', how='left')
od_df.rename(columns={'latitude': 'lat_o', 'longitude': 'lon_o'}, inplace=True)

# 重命名 village_df 的列，以便再次合并
village_df.rename(columns={'O_code': 'D_code'}, inplace=True)

# 合并 D_code 对应的经纬度
od_df = od_df.merge(village_df[['D_code', 'latitude', 'longitude']], on='D_code', how='left')
od_df.rename(columns={'latitude': 'lat_d', 'longitude': 'lon_d'}, inplace=True)

# 保存结果
od_df.to_csv('../input_data/merged_od_data_612.csv', index=False)