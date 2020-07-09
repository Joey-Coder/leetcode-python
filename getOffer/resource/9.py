class Solution:
    def minArray(self, numbers: List[int]) -> int:
        # 二分法
        lo, hi = 0, len(numbers) - 1
        while lo < hi:
            mid = ( lo + hi ) // 2
            # 如果中间点大于右边点，证明旋转点一定在[mid+1, hi]
            if numbers[mid] > numbers[hi]:
                lo = mid + 1
            # 如果中间点小于右边点，证明旋转点一定在[lo, mid]
            elif numbers[mid] < numbers[hi]:
                hi = mid
            # 如果中间点等于最右点，则无法判断，这时候消去一个重复值
            else:
                hi -= 1

        return numbers[lo]