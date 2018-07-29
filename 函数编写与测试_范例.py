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
                
if __name__ == '__main__':
    testFindRood()