import pandas as pd

def analyze_population_flows(area_list):
    # 初始化一个空的DataFrame来存储所有地区的统计数据
    columns = ['Region', 'OUT-Stronger', 'OUT-Weaker', 'OUT-SumPre', 'OUT-SumAft', 'IN-Stronger', 'IN-Weaker', 'IN-SumPre', 'IN-SumAft']
    stats_df = pd.DataFrame(columns=columns)

    for area in area_list:
        # 构建输入文件路径
        out_path = f'../input_data/{area}/{area}_out_od.csv'
        in_path = f'../input_data/{area}/{area}_in_od.csv'

        # 读取CSV文件
        df_out = pd.read_csv(out_path)
        df_in = pd.read_csv(in_path)

        # 统计满足条件的记录数
        count_strong_out = (df_out['pop_flow_pre'] < df_out['pop_flow_aft']).sum()
        count_weak_out = (df_out['pop_flow_pre'] > df_out['pop_flow_aft']).sum()

        count_strong_in = (df_in['pop_flow_pre'] < df_in['pop_flow_aft']).sum()
        count_weak_in = (df_in['pop_flow_pre'] > df_in['pop_flow_aft']).sum()

        # 计算总和
        sum_pop_flow_pre_strong = df_out['pop_flow_pre'].sum()
        sum_pop_flow_aft_strong = df_out['pop_flow_aft'].sum()

        sum_pop_flow_pre_in = df_in['pop_flow_pre'].sum()
        sum_pop_flow_aft_in = df_in['pop_flow_aft'].sum()

        # 添加统计数据行
        stats_df.loc[len(stats_df)] = [area, count_strong_out, count_weak_out, sum_pop_flow_pre_strong, sum_pop_flow_aft_strong, count_strong_in, count_weak_in, sum_pop_flow_pre_in, sum_pop_flow_aft_in]

    # 输出表格
    stats_df.to_csv('../output_data/count_pre_aft.csv',index=False)

if __name__ == "__main__":
    area_list = ['xingyang', 'jinshui', 'erqi', 'dengfeng', 'xinzheng', 'zhongmou', 'zhongyuan'] 
    analyze_population_flows(area_list)