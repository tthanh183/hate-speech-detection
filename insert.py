import pandas as pd

# Đọc file CSV gốc (có cột label, tweet)
df_original = pd.read_csv('original.csv')

# Đọc file CSV mới (có cột label, tweet)
df_new = pd.read_csv('modified_file.csv')

# Kiểm tra xem file gốc có dữ liệu bị trùng không
df_combined = pd.concat([df_original, df_new], ignore_index=True)

# Xóa các dòng bị trùng nếu cần
df_combined = df_combined.drop_duplicates().reset_index(drop=True)

# Lưu DataFrame mới vào file CSV
df_combined.to_csv('file_gop.csv', index=False)
