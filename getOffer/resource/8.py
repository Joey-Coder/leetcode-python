class Solution:
    def numWays(self, n: int) -> int:
        '对于第n级台阶，小青蛙可以从第n-1级和n-2级跳过去'
        '问题等同于斐波那契问题，初始b为n=1'
        a, b = 1, 1
        for i in range(1,n):
            a, b = b, a + b
        return b % 1000000007