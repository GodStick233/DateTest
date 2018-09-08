import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from xgboost import XGBRegressor
import xgboost as xgb

#数据预处理
data = pd.read_csv('Lyangchi1.csv')
data2 = pd.read_csv('Lyangchi1.csv')
Frequency = data.pop('Frequency')
x = data2.pop('Data')



#print(data2)

#回归预测
reg = XGBRegressor()
reg.fit(data2, data)
#joblib.dump(reg,"train.m")
reg.save_model('lyangchi1.model')

#bst2 = xgb.Booster(model_file='001.model')
#fig,ax = plt.subplots()
#fig.set_size_inches(60,30)
#xgb.plot_tree(reg,ax=ax, num_trees=2, rankdir='LR')
#fig.savefig('xgb_tree22.jpg')
#plt.show()
#tar = xgb.Booster(model_file='001.model')
#dtest = xgb.DMatrix(data2)
#preds = tar.predict(dtest)
y_pred = reg.predict(data2)

#plt.scatter(Frequency, data, s=5, label='True dates')
#plt.plot(data2,y_pred, lw=2, color='g',alpha=0.2,label='Model')
#plt.title("L-Shenmen-Train")
#plt.legend()
#plt.savefig('L_Shenmen_Train.jpg')
#plt.show()