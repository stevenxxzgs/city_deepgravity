import pandas as pd
import os

def process_od_data(input_file, point_file):
    # 读取OD数据文件
    od_data = pd.read_csv(input_file)
    od_data = od_data.dropna()  # 删除缺失值

    # 读取点文件
    point_data = pd.read_csv(point_file)
    point_codes = set(point_data['ID'])

    # 创建空DataFrame用于存储结果
    inside_records = pd.DataFrame()  # 至少一个端点在点集内的记录
    outside_records = pd.DataFrame()  # 两个端点都不在点集内的记录

    for index, row in od_data.iterrows():
        origin_code = row['O_code']
        destination_code = row['D_code']
        
        # 检查O_code或D_code是否在point_codes集合中
        if origin_code in point_codes:
            inside_records = pd.concat([inside_records, row.to_frame().T], ignore_index=True)
        elif destination_code in point_codes:
            outside_records = pd.concat([outside_records, row.to_frame().T], ignore_index=True)

    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(point_file))[0].split('_')[0]
    output_inside_path = f'../output_data/{base_name}/{base_name}_result_withoutD.csv'
    output_outside_path = f'../output_data/{base_name}/{base_name}_result_withoutO.csv'

    # 将结果写入CSV文件
    inside_records.to_csv(output_inside_path, index=False)
    outside_records.to_csv(output_outside_path, index=False)

if __name__ == "__main__":
    area_names =  ['xingyang', 'jinshui', 'erqi', 'dengfeng', 'xinzheng', 'zhongmou', 'zhongyuan'] 
    for area_name in area_names:
        input_file = f'../output_data/{area_name}/{area_name}_result.csv'
        
        point_file = f'../input_data/{area_name}/{area_name}_point.csv'

        process_od_data(input_file, point_file)