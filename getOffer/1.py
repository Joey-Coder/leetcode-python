class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        # 使用字典标记元素是否重复过
        # O(n)
        d = {}
        for x in nums:
            if d.get(x,0) != 0:
                return x
            else:
                d[x] = 1
        return -1