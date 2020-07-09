class Solution:
    def cuttingRope(self, n: int) -> int:
        """
        if n <= 3: return n - 1
        a, b = n // 3, n % 3
        if b == 0: return int(math.pow(3, a))
        if b == 1: return int(math.pow(3, a - 1) * 4)
        return int(math.pow(3, a) * 2)
        """
        """
        # 贪心算法，问题等价于可以从绳子中挑出尽量多的3，和尽量少的1
        # 优先考虑3，其次是2，但是2*2 的优先级大于3*1
        # 比如乘以5不如乘以2*3，乘以8不如乘以3*3*2
        # 2，3是最基本的单位，把一个数拆成尽量多的3和2
        if n <= 3:
            return n - 1
        res = 1
        while n > 3:
            res *= 3
            n -= 3
        # 如果最后剩下的是2，3则直接相乘
        if n == 3 or n == 2:
            return res * n
        # 如果最后为1的话，则退一个3，变换成2*2
        if n == 1:
            return res * 4 // 3
        #如果为0的话直接返回
        return res
        """

        # 解法2： 动态规划版
        if n <= 3:
            return n - 1

        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        dp[3] = 3
        for i in range(4, n + 1):
            dp[i] = max(dp[i - 3] * 3, dp[i - 2] * 2)
        return dp[n]