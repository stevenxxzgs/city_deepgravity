import geopandas as gpd
import matplotlib.pyplot as plt

# 读取Shapefile
shapefile_path = "zhengzhou_tessellation.shp"
gdf_voronoi = gpd.read_file(shapefile_path)

# 绘制地图
fig, ax = plt.subplots(figsize=(10, 10))
gdf_voronoi.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)

# 设置标题和标签
ax.set_title("Voronoi Tessellation of Villages")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 移除坐标轴
ax.axis('off')

# 保存图片
output_image_path = "./voronoi_tessellation.png"
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

print(f"图片已保存至: {output_image_path}")

# 显示图片（可选）
plt.show()