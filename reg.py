import pandas as pd
from sklearn import datasets

boston = datasets.load_boston()
#説明変数
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
#目的変数
boston_df["PRICE"] = boston.target
boston_df.head()

from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割
x_train, x_test, t_train, t_test = train_test_split(boston.data, boston.target, random_state=0)

from sklearn import linear_model

x_rm_train = x_train[:, [5]]
x_rm_test = x_test[:, [5]]

model = linear_model.LinearRegression()
model.fit(x_rm_train, t_train)


#係数
a = model.coef_
#切片
b = model.intercept_

import matplotlib.pyplot as plt

#訓練データ
plt.scatter(x_rm_train, t_train, label="Train")
#テスト用データ
plt.scatter(x_rm_test, t_test, label="Test")

y_reg = a * x_rm_train + b
plt.plot(x_rm_train, y_reg, c="red")

plt.xlabel("Room")
plt.ylabel("Price")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error

y_train  = model.predict(x_rm_train)
mse_train = mean_squared_error(t_train, y_train)
print("MSE(Train): ", mse_train)

y_test  = model.predict(x_rm_test)
mse_test = mean_squared_error(t_test, y_test)
print("MSE(Train): ", mse_test)

#線形回帰
model = linear_model.LinearRegression()
#すべての説明変数を使い学習
model.fit(x_train, t_train)

a_df = pd.DataFrame(boston.feature_names, columns=["Exp"])
a_df["a"] = pd.Series(model.coef_)
a_df

#訓練データ
y_train = model.predict(x_train)
mse_train = mean_squared_error(t_train, y_train)
print("MSE(Train): ", mse_train)

#テストデータ
y_test = model.predict(x_test)
mse_test = mean_squared_error(t_test, y_test)
print("MSE(Train): ", mse_test)