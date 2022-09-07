import pandas as pd
from openpyxl import Workbook
from sklearn import datasets

wb = Workbook()
ws = wb.active

df = pd.read_excel('HF_data.xlsx', sheet_name='2016_2019')


#説明変数

x = df[['前_入院回数(今回込）','後_入院回数']]
y = df[['死亡1']]

x1 = df[['前_入院回数(今回込）']]
x2 = df[['後_入院回数']]

print(x.shape)
print(y.shape)

from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt
import seaborn as sns

fig=plt.figure()
ax=Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

ax.scatter3D(x1, x2, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

plt.show()

""" d_df = pd.DataFrame(df.data, columns=df.feature_names)
#目的変数
d_df["A"] = df.target
d_df.head()

from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割
x_train, x_test, t_train, t_test = train_test_split(df.data, df.target, random_state=0)

from sklearn import linear_model

#線形回帰
model = linear_model.LinearRegression()
#すべての説明変数を使い学習
model.fit(x_train, t_train)

a_df = pd.DataFrame(boston.feature_names, columns=["Exp"])
a_df["a"] = pd.Series(model.coef_)
a_df

from sklearn.metrics import mean_squared_error

#訓練データ
y_train = model.predict(x_train)
mse_train = mean_squared_error(t_train, y_train)
print("MSE(Train): ", mse_train)

#テストデータ
y_test = model.predict(x_test)
mse_test = mean_squared_error(t_test, y_test)
print("MSE(Train): ", mse_test) """