import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
# 定义函数计算质心
def calculate_centroid(data):
    return np.mean(data, axis=0)

# 定义函数计算欧几里得距离
def calculate_distance(centroid1, centroid2):
    return euclidean(centroid1, centroid2)

# 示例数据集
source_domain = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 2.0],
    [1.2, 1.9]
])

target_domain = np.array([
    [4.0, 4.0],
    [4.5, 3.8],
    [5.0, 4.0],
    [4.2, 3.9]
])

# 将源域和目标域数据结合起来
combined_data = np.vstack((source_domain, target_domain))

# 使用PCA将数据映射到同一空间
pca = PCA(n_components=2)
combined_data_transformed = pca.fit_transform(combined_data)

# 分离转换后的源域和目标域数据
source_domain_transformed = combined_data_transformed[:len(source_domain)]
target_domain_transformed = combined_data_transformed[len(source_domain):]

# 计算转换后源域和目标域的质心
source_centroid = calculate_centroid(source_domain_transformed)
target_centroid = calculate_centroid(target_domain_transformed)

# 计算质心之间的距离
distance = calculate_distance(source_centroid, target_centroid)

print(f"源域质心: {source_centroid}")
print(f"目标域质心: {target_centroid}")
print(f"质心之间的距离: {distance}")

# 绘制源域和目标域的数据分布以及质心
plt.figure(figsize=(8, 6))

# 绘制源域数据点
plt.scatter(source_domain_transformed[:, 0], source_domain_transformed[:, 1], color='blue', label='源域数据')

# 绘制目标域数据点
plt.scatter(target_domain_transformed[:, 0], target_domain_transformed[:, 1], color='red', label='目标域数据')

# 绘制质心
plt.scatter(source_centroid[0], source_centroid[1], color='blue', marker='x', s=100, label='源域质心')
plt.scatter(target_centroid[0], target_centroid[1], color='red', marker='x', s=100, label='目标域质心')

# 添加质心之间的连线
plt.plot([source_centroid[0], target_centroid[0]], [source_centroid[1], target_centroid[1]], 'k--', label='质心距离')

# 添加标签和标题
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('源域与目标域数据分布及质心（经过PCA映射）')
plt.legend()
# 保存图形
plt.savefig('./Figure/PCA_domain_distribution.png')
# 显示图形
plt.show()
