import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# EDA
# 统计labels
def percentage_labels(df, label, **params):
    '''
    :param df: dataframe
    :param label: target label
    :param params: .value_counts(normalzie=True)
    :return: Series

    ex: percentage_labels(df, 'label', normalize=True)
    '''
    return df[label].value_counts(**params) * 100

# 分布图
def countplot(df, feature, title):
    sns.set()
    sns.countplot(training_data[feature])\
       .set_title(title)
    ax = plt.gca()
    for p in ax.patches:                      # 在图中加入text标注
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., # 参数参考plt.text(x, y, string)
                height + 2,
                '{:.2f}%'.format(100 * (height/df.shape[0])),
                fontsize=12, ha='center', va='bottom')
    sns.set(font_scale=1.5)
    ax.set_xlabel("Labels for {} attribute".format(feature)) # x轴名
    ax.set_ylabel("Numbers of records")  # y轴名
    plt.show()

# 各列缺失值统计条形图 barplot
# 缺失值处理需要尝试各种方法，比如第1遍用均值填充，第2遍用中位数填充，对比模型效果
def missing_barplot(df):
    x = df.columns
    y = df.isnull().sum()
    sns.set()
    sns.barplot(x, y)
    ax = plt.gca()
    for p in ax.patches:                      # 在图中加入text标注
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., # 参数参考plt.text(x, y, string)
                height + 2,
                '{:.2f}%'.format(100 * (height/df.shape[0])),
                fontsize=12, ha='center', va='bottom')
    sns.set(font_scale=1.5)
    ax.set_xlabel("Data Attributes") # x轴名
    ax.set_ylabel("Count of missing records for each attribute")  # y轴名
    plt.xticks(rotation=90)     # x 标签逆时针旋转90度
    plt.show()



# 特征相关性热图
def corrplot(df):
    sns.set()
    sns.heatmap(df.corr(),
                annot=True,   # 标注文本信息
                fmt=".1f",
    )
    plt.show()


# 特征相关性分析
# 特征关联分析：遍历拎出每个特征作target，用剩余特征来回归算法比如DecisionTreeRegressor预测，得到 R2R2  score 。
# R2R2 在0-1 之间，1表示X完美拟合y；负数表示模型拟合失败，即该特征与其他特征无相关性，无法由其他自变量推测出。
# 也就是说，该特征是必需的，because the remaining features cannot explain the variation in them.
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
dep_vars = list(train.columns)
for var in dep_vars:
    new_data = train.drop([var], axis=1)
    new_feature = pd.DataFrame(train.loc[:, var])
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.3, random_state=42)
    
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_train, y_train)
    score = dtr.score(X_test, y_test)
    print(var, 'R2 score: ',score)
    

    
    
    X_train, X_test, y_train, y_test = train_test_split(tempX, tempy, test_size=0.2, random_state=42)
    
    xgbr = xgb.XGBRegressor(random_state=42)
    xgbr.fit(X_train, y_train)
    score = xgbr.score(X_test, y_test)
    print('R2 score: ', score)
    
    model = xgbr.fit(tempX, tempy)
    return model

# 返回要预测的缺失列
def getTargetDf(df, target, other):
    target = df.loc[df[target].isnull(), other].copy()
    return target

nullmod = nullmod(df, target, other)
y210 = getTargetDf(df, target, other)
ypred = nullmod.predict(y210)
y210[target] = ypred
#-------------------------------------------
# KNN
KNeighborsClassifier(n_neighbors=5,
                     weights='uniform',
                     algorithm='auto',
                     leaf_size=30,
                     p=2,  # power, minkowski 的2次方 即euclidean_distance
                     metric='minkowski',
                     metric_params=None,
                     )


# LogisticRegression
LogisticRegression(penalty='l1',
                   dual=False,    # 如果样本数 > 特征数，则设为False
                   tol=0.0001,
                   C=1.0,
                   fit_intercept=True,
                   intercept_scaling=1, # 如果
                   class_weight=None,
                   random_state=None,
                   solver='liblinear', # liblinear 适用于小数据集
                   max_iter=100,
                   multi_class='ovr',  # 表示该问题是二分类问题
                   verbose=2,
                   )

# AdaBoost
AdaBoostClassifier(base_estimator=None,
                   n_estimators=200,
                   learning_rate=1.0,
                   )

# GradientBoosting
GradientBoostingClassifier(loss='deviance',  # 表示对分类问题 采用LogisticRegression，返回概率值
                           learning_rate=0.1,
                           n_estimators=200,
                           subsample=1.0,    # 用于调整 bias 和 variance；若<1.0，则会降方差，增偏差
                           min_samples_split=2,  # 分为一个node 要求的最小样本数
                           min_samples_leaf=1,
                           min_weight_fraction_leaf=0.0,
                           max_depth=3,
                           init=None,
                           random_state=None,
                           max_features=None,
                           verbose=0,
                           )

RandomForestClassifier(n_estimators=10,
                       criterion='gini',
                       max_depth=None,
                       min_samples_split=2,
                       min_samples_leaf=1,
                       min_weight_fraction_leaf=0.0,
                       max_features='auto',
                       max_leaf_nodes=None,
                       bootstrap=True,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0,
                       )
# -----------------------                       

    