class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        """
        def Judge(i, j):
            s = 0
            while i:
                s += i % 10
                i = i // 10
            while j:
                s += j % 10
                j = j // 10
            return s

        # 解法1： 深度优先遍历
        def DFS(i, j):
            # 递归结束条件，数组越界，i，j越界， 或者已经被遍历过了
            if not 0 <= i < m or not 0 <= j < n or Judge(i,j) > k or (i, j) in visited:
                return 0
            visited.add((i,j))
            return DFS(i+1,j) + DFS(i-1,j) + DFS(i, j-1) + DFS(i, j+1) + 1

        visited = set()

        return DFS(0,0)
        """

        # 解法2： 借助队列实现广度优先遍历
        queue = [(0, 0)]
        visited = {(0, 0)}

        def Judge(i, j):
            if 0 <= i < m and 0 <= j < n and (i, j) not in visited:
                s = 0
                while i:
                    s += i % 10
                    i = i // 10
                while j:
                    s += j % 10
                    j = j // 10
                if s <= k:
                    return True
                else:
                    return False
            return False

        res = 0
        while queue:
            node = queue.pop(0)
            res += 1
            if Judge(node[0] - 1, node[1]):
                queue.append((node[0] - 1, node[1]))
                visited.add((node[0] - 1, node[1]))
            if Judge(node[0] + 1, node[1]):
                queue.append((node[0] + 1, node[1]))
                visited.add((node[0] + 1, node[1]))
            if Judge(node[0], node[1] - 1):
                queue.append((node[0], node[1] - 1))
                visited.add((node[0], node[1] - 1))
            if Judge(node[0], node[1] + 1):
                queue.append((node[0], node[1] + 1))
                visited.add((node[0], node[1] + 1))
        return res