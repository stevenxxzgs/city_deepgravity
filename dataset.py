# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('../input_data/od_613.csv')
# df = df.dropna()  # 删除缺失值

# # 定义输出文件路径
# output_test_path = '../input_data/select_test.csv'
# output_train_path = '../input_data/select_train.csv'

# # 创建空DataFrame用于存储结果
# df_test = pd.DataFrame()
# df_train = pd.DataFrame()

# # 遍历每一行
# for index, row in df.iterrows():
#     o_code = row['O_code']
#     d_code = row['D_code']
    
#     # 检查O_code或D_code是否在指定范围内
#     if 1950 <= o_code <= 1970 or 1950 <= d_code <= 1970:
#         df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
#     else:
#         df_train = pd.concat([df_train, row.to_frame().T], ignore_index=True)

# # 将结果写入CSV文件
# df_test.to_csv(output_test_path, index=False)
# df_train.to_csv(output_train_path, index=False)



# jinshui
import pandas as pd
df = pd.read_csv('../input_data/od_613.csv')
df = df.dropna()  # 删除缺失值

# 读取包含金水区O_code的CSV文件
jinshui_df = pd.read_csv('../output_data/jinshui.csv')
jinshui_codes = set(jinshui_df['ID2'])

# 定义输出文件路径
output_test_path = '../input_data/select_test.csv'
output_train_path = '../input_data/select_train.csv'

# 创建空DataFrame用于存储结果
df_test = pd.DataFrame()
df_train = pd.DataFrame()

# 遍历每一行
for index, row in df.iterrows():
    o_code = row['O_code']
    d_code = row['D_code']
    
    # 检查O_code或D_code是否在jinshui_codes集合中
    if o_code in jinshui_codes or d_code in jinshui_codes:
        df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
    else:
        df_train = pd.concat([df_train, row.to_frame().T], ignore_index=True)

# 将结果写入CSV文件
df_test.to_csv(output_test_path, index=False)
df_train.to_csv(output_train_path, index=False)