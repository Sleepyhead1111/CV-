import numpy as np
import time


if __name__ == '__main__':
    time_start = time.perf_counter()
    a = np.arange(0, 1000, 1)
    b = np.arange(1000, 2000, 1)
    result = np.dot(a, b)
    print(f'计算结果：{result}')
    time_end = time.perf_counter()
    print(f'运行时长：{time_end - time_start}秒')

