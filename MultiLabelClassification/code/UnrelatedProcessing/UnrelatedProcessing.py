import pandas as pd

input_csv = "../../archive/train.csv"
output_csv = "../../archive/train_20kb.csv"
target_size_kb = 20  # 目标文件大小（KB）

num_rows_to_keep = 100

data = pd.read_csv(input_csv, nrows=num_rows_to_keep)
data.to_csv(output_csv, index=False)
