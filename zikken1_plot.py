import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df1 = pd.read_csv('zikken2.csv')
df2 = pd.read_csv('zikken4.csv')
# df1 = pd.read_csv('zikken3-1.csv')
# df2 = pd.read_csv('zikken3-2.csv')
# df3 = pd.read_csv('zikken3-3.csv')

# グラフの設定
plt.figure(figsize=(10, 6))

# 学習時間をプロット
plt.plot(df1['学習データ数'], df1['学習時間'], color='green', marker='o', label='Secret Sharing')
plt.plot(df2['学習データ数'], df2['学習時間'], color='blue', marker='o', label='Plaintext')
# plt.plot(df3['学習データ数'], df3['秘密分散後の正解率'], color='red', marker='o', label='epochs=3')
plt.xlabel('Number of Training Data')
plt.ylabel('time (s)')

# # 正解率をプロット
# plt.plot(df['学習データ数'], df['秘密分散前の正解率'], color='blue', marker='o', label='Accuracy Before Secret Sharing')
# plt.plot(df['学習データ数'], df['秘密分散後の正解率'], color='red', marker='o', label='Accuracy After Secret Sharing')
# plt.plot(df['学習データ数'], df['一致率'], color='green', marker='o', label='Consistency Rate')
#
# # 軸のラベルとタイトル
# plt.xlabel('Number of Training Data')
# plt.ylabel('Accuracy (%)')
# plt.title('Relationship between Number of Training Data and Accuracy')

# 凡例を表示
plt.legend()

# グリッドを表示
plt.grid(True)

# PNGファイルとして保存
plt.savefig('accuracy_graph.png')

# グラフを表示
plt.show()
