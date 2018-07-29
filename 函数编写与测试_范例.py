# 按照惯例，Python程序员使用文档字符串提供函数的“规范”。
# “规范”定义了编写者与使用者的约定，包含“假设”和“保证“—— 双方约定：“假设”xxx情况（参数）下，函数“保证”return 某结果

def findRoot(x, power, epsilon):
    """ 二分法，求开根
        x 和 epsilon 是整数或者浮点数， power 是整数。
        epsilon > 0 且 power > 1 。
        如果y ** power 和x 的差 小于epsilon，就返回浮点数y，否则返回None
    """
    if x < 0 and power % 2 == 0: # 负数没有偶数次方根
        return None
    low = min(-1.0, x)
    high = max(1.0, x)
    ans = (low + high) / 2.0
    while abs(ans ** power - x) >= epsilon:
        if ans ** power < x:
            low = ans
        else:
            high = ans
        ans = (high + low) / 2.0
    return ans
    

def testFindRood():
    epsilon = 0.0001
    for x in [0.25, -0.25, 2, -2, 8, -8]:
        for power in range(1, 4):
            print('Testing x = {} and power = {}'.format(str(x), power))
            result = findRoot(x, power, epsilon)
            if result == None:
                print('    No root')
            else:
                print('    result ** power ~= {}'.format(x))
                

# 递归
# 一般情况下，递归定义包括两部分。基本情形 和递归情形(或称归纳情形)
# 阶乘
def factR(n):
    """
        假设n 是正整数
        返回n！
    """
    if n == 1:
        return 1
    else:
        return n * factR(n -1 )



# 斐波那契数列
def fib(n):
    """
        假定n 是正整数
        返回第n 个斐波那契数
    """"
    if n == 0 or n == 1:    # 基本情形
        return 1
    else:                   # 递归情形
        return fib(n - 1) + fib(n - 2)
        
def testFib(n):
    for i in range(n+1):
        print('fib of {} = {}'.format(i, fib(i)))
        
                
if __name__ == '__main__':
    testFindRood()
    testFib(5)