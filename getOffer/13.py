class Solution:
    def cuttingRope(self, n: int) -> int:
        # 动态规划
        # 递推式：dp[i] = max(dp[i-2]*2, dp[i-3]*3), i > 3
        if n <= 3:
            return n - 1

        dp = [0] * (n+1)

        for i in range(4):
            dp[i] = i

        for i in range(4, n+1):
            dp[i] = max(dp[i-2]*2, dp[i-3]*3)

        return dp[i] % 1000000007