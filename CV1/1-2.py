import numpy as np
import time


# 数组的点积运算
def dot_product(a, b):
    if len(a) != len(b):
        print('两个数组长度不相等！')
        exit(0)
    else:
        result = 0
        for i, j in zip(a, b):
            result += i * j
        return result


if __name__ == '__main__':
    time_start = time.perf_counter()
    a = []
    b = []
    for i in range(1000):
        a.append(i)
        b.append(1000+i)
    result = dot_product(a, b)
    print(f'计算结果：{result}')
    time_end = time.perf_counter()
    print(f'运行时长：{time_end - time_start}秒')

