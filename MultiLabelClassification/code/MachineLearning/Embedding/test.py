import h5py

# 指定 HDF5 文件的路径
hdf5_filename = "../../../archive/embeddings.h5"  # 你的 HDF5 文件名

# 打开 HDF5 文件
with h5py.File(hdf5_filename, 'r') as hdf5_file:
    # 读取数据集（如果有多个数据集，可以根据数据集名称进行读取）
    dataset = hdf5_file['embeddings']

    # 将数据集的内容读取为 NumPy 数组
    embeddings = dataset[:]

print(embeddings.shape)