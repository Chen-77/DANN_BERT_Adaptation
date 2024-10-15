import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
# 定义函数计算质心
def calculate_centroid(data):
    return np.mean(data, axis=0)

# 定义函数计算欧几里得距离
def calculate_distance(centroid1, centroid2):
    return euclidean(centroid1, centroid2)

# 示例三维数据集
source_domain = np.array([
    [1.0, 2.0, 3.0],
    [1.5, 1.8, 3.2],
    [2.0, 2.0, 3.4],
    [1.2, 1.9, 3.1]
])

target_domain = np.array([
    [4.0, 4.0, 5.0],
    [4.5, 3.8, 5.1],
    [5.0, 4.0, 5.3],
    [4.2, 3.9, 5.2]
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

# 绘制三维源域和目标域的数据分布及转换后的二维数据
#fig = plt.figure(figsize=(16, 6))

# 绘制三维源域和目标域数据
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(source_domain[:, 0], source_domain[:, 1], source_domain[:, 2], color='blue', label='源域数据')
ax1.scatter(target_domain[:, 0], target_domain[:, 1], target_domain[:, 2], color='red', label='目标域数据')
ax1.set_xlabel('特征1')
ax1.set_ylabel('特征2')
ax1.set_zlabel('特征3')
ax1.set_title('源域与目标域原始三维数据分布')
ax1.legend()
plt.savefig('domain_distribution_3d.png')
plt.show()


# 绘制转换后的二维数据
fig = plt.figure(figsize=(8, 6))
ax2 = fig.add_subplot(111)
ax2.scatter(source_domain_transformed[:, 0], source_domain_transformed[:, 1], color='blue', label='源域数据')
ax2.scatter(target_domain_transformed[:, 0], target_domain_transformed[:, 1], color='red', label='目标域数据')
ax2.scatter(source_centroid[0], source_centroid[1], color='blue', marker='x', s=100, label='源域质心')
ax2.scatter(target_centroid[0], target_centroid[1], color='red', marker='x', s=100, label='目标域质心')
ax2.plot([source_centroid[0], target_centroid[0]], [source_centroid[1], target_centroid[1]], 'k--', label='质心距离')
ax2.set_xlabel('主成分1')
ax2.set_ylabel('主成分2')
ax2.set_title('源域与目标域数据分布及质心（经过PCA映射）')
ax2.legend()
plt.savefig('domain_distribution_2d.png')
plt.show()

# 保存图形
#plt.savefig('./Figure/domain_distribution_3d_to_2d.png')

# 显示图形
#plt.show()
