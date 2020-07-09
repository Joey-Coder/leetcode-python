class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        """
        本质：将每一行的数组进行遍历
        ---- 如果目标元素大于则该行元素直接break，跳到下一行
        ---- 如果匹配成功则返回该元素
        ---- 最坏情况应该为么m*n
        ---- 平均情况为 m * n / 2
        """
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] > target:
                    break
                elif matrix[i][j] == target:
                    return True
        return False