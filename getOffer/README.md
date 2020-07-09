1. 数组中的重复数字    
<span id="jump">
```python
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
```       

2. 二维数组中的查找    

```python
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
```      

3. 替换空格    

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = ''
        for c in s:
            if c == ' ':
                res += '%20'
            else:
                res += c
        return res
```    

4. [从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)     

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        # 利用栈的FILO
        stack = []
        while head:
            stack.insert(0,head.val)
            head = head.next
        
        return stack
```     

5. [重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)    

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        
        """
        本质： 递归法
        ----- 前序遍历中取出一个数为根节点
        ----- 依照该节点在中序遍历的位置将中序序列分为左子树和右子树
        ----- 依照左右子树的数量将前序遍历分为左右子树
        """
        def helper(pre: List[int], prestart: int, preend: int, ino: List[int], inostart: int, inoend: int):

            if prestart > preend:
                return None

            rootVal = pre[prestart]
            root = TreeNode(rootVal)
            if prestart == preend:
                return root
            else:
                rootIndex = ino.index(rootVal)
                leftNodes = rootIndex - inostart
                rightNodes = inoend - rootIndex
                leftSubtree = helper(pre, prestart + 1, prestart + leftNodes, ino, inostart, rootIndex - 1)
                rightSubtree = helper(pre, preend - rightNodes + 1, preend, ino, rootIndex + 1, inoend)
                root.left = leftSubtree
                root.right = rightSubtree
                return root
    
        if not preorder:
            return None
        root = helper(preorder, 0, len(preorder)-1, inorder, 0, len(inorder) - 1)
        return root
```    

6. [用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)    

```python
class CQueue:
    # 其中一个栈起到中间过渡作用
    def __init__(self):
        self.stack = []

    def appendTail(self, value: int) -> None:
        self.stack.append(value)

    def deleteHead(self) -> int:
        if not self.stack:
            return -1

        t = []
        while self.stack:
            t.append(self.stack.pop())
        
        res = t.pop()

        self.stack = []
        while t:
            self.stack.append(t.pop())
        return res
```     

7. [斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)    

```python
class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        if n == 0:
            return a
        for i in range(2, n+1):
            a, b = b , (a + b) % 1000000007
            
        return b
```    

8. [青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)    

```python
class Solution:
    def numWays(self, n: int) -> int:
        '对于第n级台阶，小青蛙可以从第n-1级和n-2级跳过去'
        '问题等同于斐波那契问题，初始b为n=1'
        a, b = 1, 1
        for i in range(1,n):
            a, b = b, a + b
        return b % 1000000007
```    

9. [旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)   

```python
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
```

10. [矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)   

```python
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
```     

11. [机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)   

```python
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
        queue = [(0,0)]
        visited = {(0,0)}
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
            if Judge(node[0]-1, node[1]):
                queue.append((node[0]-1, node[1]))
                visited.add((node[0]-1, node[1]))
            if Judge(node[0]+1, node[1]):
                queue.append((node[0]+1, node[1]))
                visited.add((node[0]+1, node[1]))
            if Judge(node[0], node[1]-1):
                queue.append((node[0],node[1]-1))
                visited.add((node[0], node[1]-1))
            if Judge(node[0], node[1]+1):
                queue.append((node[0], node[1]+1))
                visited.add((node[0], node[1]+1))
        return res
```    

12. [剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)    

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        """
        if n <= 3: return n - 1
        a, b = n // 3, n % 3
        if b == 0: return int(math.pow(3, a))
        if b == 1: return int(math.pow(3, a - 1) * 4)
        return int(math.pow(3, a) * 2)
        """
        """
        # 贪心算法，问题等价于可以从绳子中挑出尽量多的3，和尽量少的1
        # 优先考虑3，其次是2，但是2*2 的优先级大于3*1
        # 比如乘以5不如乘以2*3，乘以8不如乘以3*3*2
        # 2，3是最基本的单位，把一个数拆成尽量多的3和2
        if n <= 3:
            return n - 1
        res = 1
        while n > 3:
            res *= 3
            n -= 3
        # 如果最后剩下的是2，3则直接相乘
        if n == 3 or n == 2:
            return res * n
        # 如果最后为1的话，则退一个3，变换成2*2
        if n == 1:
            return res * 4 // 3
        #如果为0的话直接返回
        return res
        """

        # 解法2： 动态规划版
        if n <= 3:
            return n - 1
        
        dp = [0] * (n+1)
        dp[1] = 1
        dp[2] = 2
        dp[3] = 3
        for i in range(4,n+1):
            dp[i] = max(dp[i-3]*3, dp[i-2]*2)
        return dp[n]
```    

13. [剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)   

```python
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
```     

14. [二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)    

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
       # 按位操作
        res = 0
        while n:
            res += n & 1
            n = n >> 1
        return res
```    

15. [数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)      

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 二分法，相比于将n个x相乘的方法，复杂度降对数级
        # x ** n = (x ** (n//2)) ** 2 or (x ** (n//2)) ** 2 * x
        # 3 ** 6 = (3 ** 3) ** 2 = ((3 ** 1) ** 2 * 3) ** 2
        # 当x为零时直接返回，避免因为-x/0次放报错
        if x == 0:
            return 0
        res = 1
        if n < 0:
            x, n = 1 / x, -n
        while n:
            # 与1做与操作，相当于判断奇偶
            # 如果是奇数，则需要先乘以x
            if n & 1:
                res *= x
            x *= x
            # 相当于//2
            n >>= 1
        return res
```    

16. [打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)    

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return [x for x in range(1, 10 ** n)]
```    

17. [删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)   

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        # 找到要删除节点的前去节点
        pre = head
        cur = head.next
        if pre.val == val:
            head = head.next
        else:
            while cur:
                if cur.val == val:
                    pre.next = cur.next
                    break
                pre, cur = cur, cur.next
        return head
```    

18. [正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)    

```python
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
        dp = [[False for _ in range(m+1)] for _ in range(n+1) ]

        for i in range(n+1):
            for j in range(m+1):
                
                if j == 0:
                    if i == 0:
                        dp[i][j] = True
                else:
                    if p[j-1] != '*':
                        if i > 0 and (s[i-1] == p[j-1] or p[j-1] == '.'):
                            dp[i][j] = dp[i-1][j-1]
                    else:
                        # * 为 0
                        if j >= 2:
                            dp[i][j] = dp[i][j-2]
                        # * 大于 0
                        if j >=2 and i >= 1 and (s[i-1] == p[j-2] or p[j-2] =='.'):
                            dp[i][j] = dp[i][j] or dp[i-1][j]                    
        return dp[n][m]
```     

19. [表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)   

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        """
        s = s.strip()
        if not s:
            return False
        numSeen = False
        dotSeen = False
        eSeen = False
        for i in range(0,len(s)):
            if s[i].isdigit():
                numSeen = True
            elif s[i] == '.':
                # .之前不能出现.和e
                if dotSeen or eSeen:
                    return False
                dotSeen = True
            elif s[i] == 'e':
                # e之前不能出现e
                if eSeen or not numSeen:
                    return False
                eSeen = True
                # e之后必须出现数字
                numSeen = False
            elif s[i] in {'+', '-'}:
                # +，-必须在第一位或者e的后面
                if i != 0 and s[i-1] != 'e':
                    return False
            else:
                return False
        return numSeen
        """
        # 解法二：正则
        p = re.compile(r'^[+-]?(\.\d+|\d+\.?\d*)([eE][+-]?\d+)?$')
        return p.match(s.strip()) != None
```    

20. [调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)    

```python
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
```
