import time


# 斐波那契数列
def Fibonacci(n):
    if n <= 0:
        print('输入值异常，请输入一个正整数！')
        exit(0)
    else:
        list = []
        list.append(1)
        a = 1
        b = 0
        for i in range(1, n):
            a, b = a+b, a
            list.append(a)
    print(list)


if __name__ == '__main__':
    n = eval(input('请输入需要显示斐波那契数列的前几项：'))
    time_start = time.perf_counter()
    Fibonacci(n)
    time_end = time.perf_counter()
    print(f'运行时长：{time_end - time_start}秒')
