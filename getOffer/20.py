class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        l = len(nums)
        i = j = 0
        while j < l:
            if nums[j] & 1:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
            j += 1
        return nums