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
    # 从TXT文件读取边界点
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    boundary_points = [list(map(float, line.strip().split(','))) for line in lines]
    return boundary_points

def generate_output_filenames(base_name):
    # 生成输出文件名
    base_name = base_name.replace('.txt', '')
    ensure_directory_exists(f'../input_data/{base_name}')
    output_file = f'../input_data/{base_name}/{base_name}_point.csv'
    output_html = f'../output_data/{base_name}_output_map.html'
    return output_file, output_html

def filter_and_plot_points(input_file, boundary_points, output_file, output_html):
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
    df_polygon_output.to_csv(output_file, index=False)

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

if __name__ == "__main__":
    txt_file = './config/xingyang.txt'
    input_file = '../input_data/Villages_lat_lon.csv'
    boundary_points = read_boundary_points_from_txt(txt_file)
    output_file, output_html = generate_output_filenames(txt_file.split('/')[-1])
    filter_and_plot_points(input_file, boundary_points, output_file, output_html)