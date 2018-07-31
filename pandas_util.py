#-----------pandas 常用代码
# 用字典 创建DataFrame
pd.DataFrame.from_dict(dict, orient='index').T

X = X.loc[(X < 0).sum(1) == 0]  # 不含缺失数据的行（如-1， -2 值）
df.loc[(df > 0.2).any(1)]

gb = df.groupby('label')  # 字典：key是label的值，group是对应的组
gb.get_group(1)  # label=1
for name, group in gb:
    pos = group[group['y'] == 1]
    neg = group[group['y'] == 0]
    
# 最快筛选并替换值
y = 1 * (df.cand_pty_affiliation == 'REP') # 将 label = 'REP' 的改为1，否则为0

# 删除只有1个值的列
def drop1(df):
    onecols = df.columns[df.std() == 0]
    df.drop(onecols, axis=1, inplace=True)
    return df
    
# pandas 处理文本数据
s.str.split(',')
s.str.extract？

# pandas 批量更改列名 .columns.get_level_values
f4.columns = f4.columns.get_level_values(0) + 'ctime_'

# 删除重复的列

X.T.drop_dumplicates().T
# 若发送递归深度报错，则执行：
import sys
sys.setrecursionlimit(15000) # 15000可以修改