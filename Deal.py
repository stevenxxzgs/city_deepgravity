import pandas as pd
import folium
from shapely.geometry import Polygon, Point
from folium.plugins import MarkerCluster
import os

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_boundary_points_from_txt(txt_file):
    """从TXT文件读取边界点"""
    boundary_points = []
    with open(txt_file, 'r') as file:
        for line in file:
            # 去除行尾的换行符，并去除行首尾的空白字符
            line = line.strip()
            # 如果行非空，则继续处理
            if line:
                coords = list(map(float, line.split(',')))
                boundary_points.append(coords)
    return boundary_points

def generate_output_filenames(base_name):
    """生成输出文件名"""
    base_name = base_name.replace('.txt', '')
    ensure_directory_exists(f'../input_data/{base_name}')
    output_point_csv = f'../input_data/{base_name}/{base_name}_point.csv'
    output_html = f'../output_data/{base_name}_output_map.html'
    return output_point_csv, output_html

def filter_and_plot_points(input_file, boundary_points, output_point_csv, output_html):
    """过滤并绘制点"""
    df = pd.read_csv(input_file)
    boundary_polygon = Polygon(boundary_points)

    def is_point_inside_polygon(row):
        point = Point(row['latitude'], row['longitude'])
        return boundary_polygon.contains(point)

    df_polygon = df[df.apply(is_point_inside_polygon, axis=1)]

    if df_polygon.empty:
        print("No valid points found within the specified boundary.")
        return

    df_polygon_output = df_polygon['ID']
    df_polygon_output.to_csv(output_point_csv, index=False)

    center_lat = df_polygon['latitude'].mean()
    center_lon = df_polygon['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    marker_cluster = MarkerCluster().add_to(m)

    for index, row in df_polygon.iterrows():
        lat, lon = row['latitude'], row['longitude']
        folium.Marker(
            location=[lat, lon],
            popup=f"Latitude: {lat}, Longitude: {lon}",
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    folium.Polygon(
        locations=boundary_points,
        color='red',
        fill=False,
        weight=2
    ).add_to(m)

    m.save(output_html)

def process_od_data(input_file, point_file, area_name):
    """处理OD数据"""
    # 读取OD数据文件
    od_data = pd.read_csv(input_file)
    od_data = od_data.dropna()  # 删除缺失值

    # 读取点文件
    point_data = pd.read_csv(point_file)
    point_codes = set(point_data['ID'])

    # 创建空DataFrame用于存储结果
    inside_records = pd.DataFrame()  # 至少一个端点在点集内的记录
    outside_records = pd.DataFrame()  # 两个端点都不在点集内的记录
    test_records = pd.DataFrame()
    train_records = pd.DataFrame()

    for index, row in od_data.iterrows():
        origin_code = row['O_code']
        destination_code = row['D_code']
        
        # 检查O_code或D_code是否在point_codes集合中
        if origin_code not in point_codes and destination_code in point_codes:
            inside_records = pd.concat([inside_records, row.to_frame().T], ignore_index=True)
        if origin_code in point_codes and destination_code not in point_codes:
            outside_records = pd.concat([outside_records, row.to_frame().T], ignore_index=True)
        if origin_code in point_codes or destination_code in point_codes:
            test_records = pd.concat([test_records, row.to_frame().T], ignore_index=True)
        if origin_code not in point_codes and destination_code not in point_codes:
            train_records = pd.concat([train_records, row.to_frame().T], ignore_index=True)

    # 生成输出文件名
    output_inside_path = f'../input_data/{area_name}/{area_name}_in_od.csv'
    output_outside_path = f'../input_data/{area_name}/{area_name}_out_od.csv'
    test_records_path = f'../input_data/{area_name}/{area_name}_od.csv'
    train_records_path = f'../input_data/{area_name}/without_{area_name}_od.csv'

    # 将结果写入CSV文件
    inside_records.to_csv(output_inside_path, index=False)
    outside_records.to_csv(output_outside_path, index=False)
    test_records.to_csv(test_records_path, index=False)
    train_records.to_csv(train_records_path, index=False)

if __name__ == "__main__":
    input_file = '../input_data/Villages_lat_lon.csv'
    input_od_file = '../input_data/od_613.csv'
    area_names = ['xingyang', 'jinshui', 'erqi', 'xinmi', 'dengfeng', 'xinzheng', 'zhongmou', 'zhongyuan'] 
    for area_name in area_names:
        txt_file = f'./config/{area_name}.txt'

        # 读取边界点
        boundary_points = read_boundary_points_from_txt(txt_file)

        # 生成输出文件名
        output_point_csv, output_html = generate_output_filenames(txt_file.split('/')[-1])

        # 过滤并绘制点
        filter_and_plot_points(input_file, boundary_points, output_point_csv, output_html)

        # 处理OD数据
        process_od_data(input_od_file, output_point_csv, area_name)