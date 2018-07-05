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


# 数据类型转换：减少内存
floatcols = df.select_dtypes(include=['float64']).columns
intcols = []
fltcols = []
for col in floatcols:
    if np.modf(df[col])[0].sum() == 0:
        intcols.append(col)
        df[col] = df.astype(np.int16, errors='ignore')
    else:
        fltcols.append(col)

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
    
# 与某单变量相关的所有变量：    
def dependent(feature, df, gt=0.6):
	dependence = []
	for col in df.columns:
		corrn = df[feature].corr(df[col])
		if corrn >= gt:
			dependence.append((col, corrn))
	d = pd.DataFrame(dependence, columns=['features','coef'])
	return d


# 特征工程-------------------

#纠正偏态分布，一些特征值更可能自然分布，所以用自然对数转换
log_data = np.log(data + 1)
pd.plotting.scatter_matrix(log_data, alpha=0.3, figsize=(14, 8), diagonal='kde')

import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
X_train_1 = np.array(X_train)
X_train_transform = transformer.transform(X_train_1)

# 用其他features 回归预测缺失值     
# 返回预测模型          
def nullmod(df, target, other):
    """
    df: Xcomplete
    target: 'f210'
    other: ['f110','f209','f106','f81','f105','f73','f75','f109','f208','f74','f72','f80','f104']
    """
    tempX = df.loc[df[target].notnull(), other]
    tempy = df.loc[df[target].notnull(), target]

    
    
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
# 基于投票的集成方法 Voting-based ensenmble ML model
# 先将baseline 做CV 优化得到best，在集成
from sklearn.ensemble import VotingClassifier
votingMod = VotingClassifier(estimators=[('gb', bestGbModFitted_transformed),  
                                         ('ada', bestAdaModFitted_transformed)], 
                             voting='soft',weights=[2,1])
votingMod = votingMod.fit(X_train_transform, y_train)

#----------- 调参  
# GridSearchCV  VS RandomizedSearchCV  https://blog.csdn.net/juezhanangle/article/details/80051256
# GridSearchCV 即网格搜索和交叉验证，
# 可以保证在指定的参数范围内找到精度最高的参数，但是这也是网格搜索的缺陷所在，它要求遍历所有可能参数的组合，
# 在面对大数据集和多参数的情况下，非常耗时；
# 再加上CV，更加耗时：
· 将训练数据集划分为K份，K一般为10
· 依次取其中一份为验证集，其余为训练集训练分类器，测试分类器在验证集上的精度 
· 取K次实验的平均精度为该分类器的平均精度

# RandomizedSearchCV
以在参数空间中随机采样 代替 网格遍历搜索，在对于有连续变量的参数时，将其当作一个分布进行采样，它的搜索能力取决于设定的n_iter参数。
一般建议使用 RandomizedSearchCV，尤其大规模数据集上的集成算法。


# 二者用法一致，代码如下
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
 
clf1 = xgb.XGBClassifier()

param_dist = {
        'n_estimators':range(80,200,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1)
        }

grid = GridSearchCV(clf1,                      # 训练器
                    param_dist,                # 参数空间，字典类型
                    cv = 3,   
                    scoring = 'neg_log_loss',
                    n_iter=300,                # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
                    n_jobs = -1
                    )
 

grid.fit(traindata.values,np.ravel(trainlabel.values)) #在训练集上训练

best_estimator = grid.best_estimator_  #返回最优的训练器
print(best_estimator)

print(grid.best_score_)  #输出最优训练器的精度

# 自定义评分参数 scoring:
import numpy as np
from sklearn.metrics import make_scorer
 
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll
 
#这里的greater_is_better参数决定了自定义的评价指标是越大越好还是越小越好
loss  = make_scorer(logloss, greater_is_better=False)
score = make_scorer(logloss, greater_is_better=True)




    