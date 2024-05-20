import argparse


def isosceles_triangle(h):
    for i in range(1, h+1):
        print(' ' * (h-i), f'{i}' * (2*i-1), ' ' * (h-i), sep='', end='\n')


if __name__ == '__main__':
    # 用来装载参数的容器
    parser = argparse.ArgumentParser(description='isosceles triangle')
    # 给这个解析对象添加命令行参数
    parser.add_argument('--height', type=int, choices=range(1, 10), default=5, metavar='', help='Height of isosceles triangle')
    # 获取所有参数
    args = parser.parse_args()
    # 绘制等腰三角形
    isosceles_triangle(args.height)

