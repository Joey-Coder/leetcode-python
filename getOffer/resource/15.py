class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 二分法，相比于将n个x相乘的方法，复杂度降对数级
        # x ** n = (x ** (n//2)) ** 2 or (x ** (n//2)) ** 2 * x
        # 3 ** 6 = (3 ** 3) ** 2 = ((3 ** 1) ** 2 * 3) ** 2
        # 当x为零时直接返回，避免因为-x/0次放报错
        if x == 0:
            return 0
        res = 1
        if n < 0:
            x, n = 1 / x, -n
        while n:
            # 与1做与操作，相当于判断奇偶
            # 如果是奇数，则需要先乘以x
            if n & 1:
                res *= x
            x *= x
            # 相当于//2
            n >>= 1
        return res