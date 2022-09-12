import pandas as pd
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

df = pd.read_excel('コピー.xlsx', sheet_name='2016_2019')

#説明変数

x = df[['年齢','前_入院回数(今回込）','後_入院回数','高脂血症','心不全入院1']]
""" x = df[['前_入院回数(今回込）','後_入院回数']] """
y = df[['死亡1']]

from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割
x_train, x_test= train_test_split(x)
y_train, y_test= train_test_split(y)



print(x_train)
""" x1 = x_train[['前_入院回数(今回込）']]
x2 = x_train[['後_入院回数']]

print(x.shape)
print(y.shape)

from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt

fig=plt.figure()
ax=Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

ax.scatter3D(x1, x2, y_train)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

plt.show() """


from sklearn import linear_model

#線形回帰
model = linear_model.LinearRegression()
#すべての説明変数を使い学習
model.fit(x_train, y_train)



from sklearn.metrics import mean_squared_error

#訓練データ
out_train = model.predict(x_train)
mse_train = mean_squared_error(y_train, out_train)
print("MSE(Train): ", mse_train)

#テストデータ
out_test = model.predict(x_test)
mse_test = mean_squared_error(y_test, out_test)
print("MSE(Test): ", mse_test)