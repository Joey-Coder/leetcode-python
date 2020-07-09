class Solution:
    def hammingWeight(self, n: int) -> int:
       # 按位操作
        res = 0
        while n:
            res += n & 1
            n = n >> 1
        return res