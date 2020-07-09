class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # 深度优先遍历 + 剪枝
        def dfs(i, j, k):
            # 递归终止条件：i，j越界，或者成功匹配word即k = len(word) - 1
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            # 剪枝操作，标记已经走过的路径为“/”，防止再次遍历
            tmp, board[i][j] = board[i][j], '/'
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            board[i][j] = tmp
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                     return True
        return False