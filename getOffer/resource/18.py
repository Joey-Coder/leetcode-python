class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # 动态规划，自底向上求解
        # dp[i][j]表示使用s[:i]和p[:j]是否匹配
        # 初始状态：dp[0][0] = True,dp[1][0]..,dp[n][0] =False
        # 递推表达式：dp[i][j] = dp[i-1]dp[j-1] (i>0 and j>0 and s[i]=p[j] or p[j] = '.' )
        # dp[i][j] = dp[i][j-2](j >= 2 and p[j-1] = * )
        # dp[i][j] = dp[i-1][j] (j >= 2 and p[j-1] = * and (p[j-2] != . or p[j-2] == s[i-1]))
        # 如果p[j] = *时，有两种可能，一种是*=0，这是一定可以的，对应的表达式就是dp[i][j] = dp[i][j-2]
        # 但是如果在上面的条件下，p[j-2] = .或者p[j-2] = s[i-1],则*也可以表达为1，2，3...所以*表达为0还是大于0取决于，表达之后dp[i][j]是否为真dp[i][j] = dp[i][j-2] or dp[i-1]dp[j]
        n = len(s)
        m = len(p)
        dp = [[False for _ in range(m + 1)] for _ in range(n + 1)]

        for i in range(n + 1):
            for j in range(m + 1):

                if j == 0:
                    if i == 0:
                        dp[i][j] = True
                else:
                    if p[j - 1] != '*':
                        if i > 0 and (s[i - 1] == p[j - 1] or p[j - 1] == '.'):
                            dp[i][j] = dp[i - 1][j - 1]
                    else:
                        # * 为 0
                        if j >= 2:
                            dp[i][j] = dp[i][j - 2]
                        # * 大于 0
                        if j >= 2 and i >= 1 and (s[i - 1] == p[j - 2] or p[j - 2] == '.'):
                            dp[i][j] = dp[i][j] or dp[i - 1][j]
        return dp[n][m]