<span id="jump1"></span>

1. 数组中的重复数字   
 
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

<span id="jump2"></span>

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

<span id="jump3"></span>

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

<span id="jump4"></span>

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

<span id="jump5"></span>

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

<span id="jump6"></span>

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

<span id="jump7"></span>

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

<span id="jump8"></span>

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

<span id="jump9"></span>   

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

<span id="jump10"></span>

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

<span id="jump11"></span>

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

<span id="jump12"></span>

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

<span id="jump13"></span>

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

<span id="jump14"></span>

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

<span id="jump15"></span>

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

<span id="jump16"></span>

16. [打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)    

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return [x for x in range(1, 10 ** n)]
```    

<span id="jump17"></span>

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

<span id="jump18"></span>

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

<span id="jump19"></span>

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

<span id="jump20"></span>

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

<span id="jump21"></span>

21. [链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)   

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        l, node = 0, head
        while node:
            l += 1
            node = node.next
        while k < l:
            head = head.next
            k += 1
        return head
```     

<span id="jump22"></span>

22. [反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)   

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return
        pre = head
        cur = head.next
        pre.next = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre, cur = cur, tmp
        return pre
```     

<span id="jump23"></span>

23. [合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)    

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        newlist = head = ListNode(-1)
        while l1 or l2:
            if not l1:
                head.next = ListNode(l2.val)
                l2 = l2.next
            elif not l2:
                head.next = ListNode(l1.val)
                l1 = l1.next
            elif l1.val <= l2.val:
                head.next = ListNode(l1.val)
                l1 = l1.next
            else:
                head.next = ListNode(l2.val)
                l2 = l2.next
            head = head.next
        return newlist.next
```     

<span id="jump24"></span>

24. [树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)   

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool: 
        # 递归判断
        # 如果t为空，则证明子树被s覆盖返回True
        # 如果s为空，且t不为空，证明s被t覆盖，返回False
        # 如果s，t均不为空，且相等，返回False
        # 递归判别，s的左子树和t的左子树，s的右子树和t的右子树   
        def Judge(s, t):
            if not t:
                return True
            else:
                if not s or s.val != t.val:
                    return False
            left = Judge(s.left, t.left)
            right = Judge(s.right, t.right)
            return left and right

        # 深度优先(前序遍历)
        if not A or not B:
            return False
        stack = [A]
        res = False
        while stack:
            node = stack.pop()
            if node.val == B.val:
                res = Judge(node, B)
                if res:
                    return res
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
```    

<span id="jump25"></span>

25. [二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)      

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        node = TreeNode(root.val)
        left = self.mirrorTree(root.left)
        right = self.mirrorTree(root.right)
        # 这里如果把node.left, node.right = left, right就可以变成拷贝二叉树了
        node.left, node.right = right, left
        return node
        # 本题也可以用栈来实现，只要在入栈的时候，交换左右节点就可以了
```    

<span id="jump26"></span>

26. [对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)     

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        # 由上至下递归，判断对称位置上的节点值是否相等
        def helper(l, r):
            if not l and not r:
                return True
            if not l or not r or l.val != r.val:
                return False
            return helper(l.left, r.right) and helper(l.right, r.left)
        return helper(root.left, root.right) if root else True
```    

<span id="jump27"></span>

27. [顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```python
class Solution:
    def spiralOrder(self, matrix:[[int]]) -> [int]:
        if not matrix: return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
            # right
            for i in range(l, r + 1): res.append(matrix[t][i]) # left to right
            t += 1
            if t > b: break
            # down
            for i in range(t, b + 1): res.append(matrix[i][r]) # top to bottom
            r -= 1
            if l > r: break
            # left
            for i in range(r, l - 1, -1): res.append(matrix[b][i]) # right to left
            b -= 1
            if t > b: break
            # up
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # bottom to top
            l += 1
            if l > r: break
        return res
```    

<span id="jump28"></span>   

28. [包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)    

```python
class MinStack:
    # 构造出一个辅助栈，用来记录输入的最小值的变化
    def __init__(self):
        self.A, self.B = [], []

    def push(self, x: int) -> None:
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self) -> None:
        if self.A.pop() == self.B[-1]:
            self.B.pop()

    def top(self) -> int:
        return self.A[-1]

    def min(self) -> int:
        return self.B[-1]
```  

<span id="jump29"></span>

29. [栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)   

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        # 使用一个辅助栈模拟出入栈的过程
        # 先遍历push栈，如果push栈的当前元素不等于poped栈顶元素，则押入辅助栈中
        # 如果相等，则stack pop，i+=1
        stack, i = [], 0
        for x in pushed:
            # num 入栈
            stack.append(x)
            # 循环判断与出栈
            while stack and stack[-1] == popped[i]: 
                stack.pop()
                i += 1
        return not stack
```    

<span id="jump30"></span>   

30. [从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return []
        queue = [root]
        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return res
```     

<span id="jump31"></span>

31. [从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = [[root]]
        res = [[root.val]]
        while queue:
            tq = []
            tmp = []
            node = queue.pop(0)
            for n in node:
                if n.left:
                    tq.append(n.left)
                    tmp.append(n.left.val)
                if n.right:
                    tq.append(n.right)
                    tmp.append(n.right.val)
            if tq:
                queue.append(tq)
            if tmp:
                res.append(tmp)
        return res
```     

<span id="jump32"></span>   

32. [从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = [[root]]
        res = [[root.val]]
        flag = False
        while queue:
            node = queue.pop(0)
            tmp = []
            tq = []
            for n in node:
                if n.left:
                    tq.append(n.left)
                    if flag:
                        tmp.append(n.left.val)
                    else:
                        tmp.insert(0,n.left.val)
                if n.right:
                    tq.append(n.right)
                    if flag:
                        tmp.append(n.right.val)
                    else:
                        tmp.insert(0, n.right.val)
            flag = not flag
            if tmp:
                res.append(tmp)
            if tq:
                queue.append(tq)
        return res
```     

<span id="jump33"></span>

33. [二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

```python
class Solution:
    def verifyPostorder(self, postorder: [int]) -> bool:
        def recur(i, j):
            # 任意两个节点都可以构成搜索树的后序序列
            if i >= j - 1:
                return True
            p = i
            # 查找左子树，左子树的所有节点小于根节点
            while postorder[p] < postorder[j]:
                p += 1
            m = p
            # 查找右子树，右子树的所有节点大于根节点
            while postorder[p] > postorder[j]:
                p += 1
            # p == j保证右子树没有小于根节点的元素
            # 递归判断左右子树
            return p == j and recur(i, m - 1) and recur(m, j - 1)

        return recur(0, len(postorder) - 1)
```    

<span id="jump34"></span>   

34. [二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)    

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        res, path = [], []
        # 先序遍历，使用栈来存储遍历的路径
        def recur(root, tar):
            if not root: return
            path.append(root.val)
            tar -= root.val
            if tar == 0 and not root.left and not root.right:
                res.append(list(path))
            recur(root.left, tar)
            recur(root.right, tar)
            # 遍历回退的时候清除之前的路径
            path.pop()
        recur(root, sum)
        return res
```     

<span id="jump35"></span>    

35. [复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)    

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # 深度优先，结束条件，节点为空，或者节点已经复制过了，直接返回节点就行了
        def DFS(head):
            if not head:
                return
            if head in visited:
                return visited[head]
            copy = Node(head.val, head.next, head.random)
            visited[head] = copy
            copy.next = DFS(head.next)
            copy.random = DFS(head.random)
            return copy
        visited = {}
        return DFS(head)
```     

<span id="jump36"></span>    

36. [叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return
        stack = []
        pre = None
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                if not pre:
                    head = node
                else:
                    pre.right = node
                node.left = pre
                pre = node
                root = node.right
        head.left, node.right = node, head
        return head
```    

<span id="jump37"></span>

37. [序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        if not root:
            return ''
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node:
                res.append(str(node.val))
                queue.extend([node.left, node.right])
            else:
                res.append('/')
        res.insert(0, '/')
        i = len(res) - 1
        while res[i] == '/':
            i -= 1
        return '[' + ','.join(res[:i+1]) + ']'
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return
        print(data)
        #dataToList = re.findall('[+-]?[0-9]+|/',data)
        dataToList = data[1:-1].split(',')
        print(dataToList)
        l = len(dataToList)
        visited = {}
        c = 1
        for i in range(1,l):
            if dataToList[i] != '/':
                if visited.get(i,-1) == -1:
                    node = TreeNode(dataToList[i])
                    visited[i] = node 
                else:
                    node = visited.get(i)
                # left child node
                if c+1 < l and dataToList[c+1] != '/':
                    left = visited.get(c+1,-1)
                    if left != -1:
                        node.left = left
                    else:
                        node.left = TreeNode(dataToList[c+1])
                        visited[c+1] = node.left
                # right child node
                if c+2 < l and dataToList[c+2] != '/':    
                    right = visited.get(c+2,-1)
                    if right != -1:
                        node.right = right
                    else:
                        node.right = TreeNode(dataToList[c+2])
                        visited[c+2] = node.right
                c += 2
            if c >= l:
                break
        return visited[1]
                

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```    

<span id="jump38"></span>   

38. [字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        c, res = list(s), []
        def dfs(x):
            if x == len(c) - 1:
                res.append(''.join(c)) # 添加排列方案
                return
            dic = set()
            for i in range(x, len(c)):
                if c[i] in dic: continue # 重复，因此剪枝
                dic.add(c[i])
                c[i], c[x] = c[x], c[i] # 交换，将 c[i] 固定在第 x 位
                dfs(x + 1) # 开启固定第 x + 1 位字符
                c[i], c[x] = c[x], c[i] # 恢复交换
        dfs(0)
        return res
```    

<span id="jump39"></span>

39. [数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)   

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        # hash法
        d = {}
        l = len(nums)
        for i in range(l):
            d[nums[i]] = d.get(nums[i],0) + 1
            if d[nums[i]] > l // 2:
                return nums[i]
        """
        # 因为存在一个元素出现的次数超过数组的一半。
        # 所以将任意两个不同的元素进行相消
        # 任意相同的元素次数累计
        t = 0
        for num in nums:
            if t == 0:
                x = num
            if num == x:
                t += 1
            else:
                t -= 1
        return x
```     

<span id="jump40"></span>   

40. [最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)    

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        arr.sort()
        return arr[:k]
```

<span id="jump41"></span>

41. [数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

```python
from heapq import *

class MedianFinder:
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0
```    

<span id="jump42"></span>

42. [连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 动态规划，递推关系式：
        # dp[i]代表以i为位置结尾的子数组的最大和
        # dp[i] = dp[i-1] + nums[i] or nums[i](if dp[i-1] < 0)
        """
        l = len(nums)
        dp = [0] * l
        for i in range(l):
            if i == 0:
                dp[0] = nums[0]
            else:
                if dp[i-1] < 0:
                    dp[i] = nums[i]
                else:
                    dp[i]  = nums[i] + dp[i-1]
        return max(dp)
        """
        # 空间优化版，可以将nums用作dp
        for i in range(1,len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
        return max(nums)
```    

<span id="jump43"></span>

43. [1～n整数中1出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)   

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit, res = 1, 0
        high, cur, low = n // 10, n % 10, 0
        while high != 0 or cur != 0:
            if cur == 0: res += high * digit
            elif cur == 1: res += high * digit + low + 1
            else: res += (high + 1) * digit
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res
```    

<span id="jump44"></span>

44. [数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        start, digit, count = 1, 1, 9
        # 找到n是几位数
        while n > count:
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit
        # 找到n是几位数中的具体哪个数。
        # n-1是因为你是从1开始属的，start已经包含了第一位
        num = start + (n - 1) // digit
        # 找到n是具体哪个数里面的具体哪一位
        res = int(str(num)[(n-1) % digit])
        return res
```    

<span id="jump45"></span>

45. [把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def sortRule(x, y):
            a, b = x + y, y + x
            if a > b:
                return 1
            elif a < b:
                return -1
            else:
                return 0

        strs = [str(num) for num in nums]
        strs.sort(key=functools.cmp_to_key(sortRule))
        return ''.join(strs)
        """
        # 快速排序
        def quickSort(lo, hi):
            if lo >= hi:
                return
            i, j = lo, hi
            # 自定义排序规则，将两个字符前后拼接，比较大小
            while i < j:
                while strs[j] + strs[lo] >= strs[lo] + strs[j] and i < j:
                    j -= 1
                while strs[i] + strs[lo] <= strs[lo] + strs[i] and i < j:
                    i += 1
                strs[i], strs[j] = strs[j], strs[i]
            strs[i], strs[lo] = strs[lo], strs[i]
            quickSort(lo, i - 1)
            quickSort(i + 1, hi)

        strs = [str(num) for num in nums]
        quickSort(0, len(nums) - 1)
        return ''.join(strs)
        """
```    

<span id="jump46"></span>

46. [把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

```python
class Solution:
    def translateNum(self, num: int) -> int:
        # 动态规划
        a = str(num)
        l = len(a)
        dp = [0] * l
        dp[0] = 1
        for i in range(1, l):
            if '10' <= a[i-1]+a[i] < '26':
                #  状态转移方程
                if i - 2 >= 0:
                    dp[i] = dp[i - 1] + dp[i - 2]
                else:
                    dp[i] = dp[i - 1] + 1
            else:
                dp[i] = dp[i - 1]
        return dp[l - 1]
```    

<span id="jump47"></span>

47. [礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        # 动态规划
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                top = 0 if i == 0 else grid[i-1][j]        
                left = 0 if j == 0 else grid[i][j-1]
                grid[i][j] += max(top, left)
        return grid[-1][-1]
        """
        # dfs 惨不忍睹
        def dfs(i, j):
            if i < 0 or j < 0:
                return 0
            if visited.get((i,j),False) != False:
                return visited[(i,j)]
            grid[i][j] += max(dfs(i-1,j), dfs(i,j-1))
            visited[(i,j)] = grid[i][j]
            return grid[i][j]
        lr = len(grid)
        lc = len(grid[0])
        visited = {}
        return dfs(lr-1, lc-1)
        """
```    

<span id="jump48"></span>

48. [最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        d, start = {}, 0
        res, disc = 0, 0
        for end in range(len(s)):
            index = d.get(s[end], -1)         
            if index < start:
                disc += 1
            else:
                disc = end - index
                start = index + 1      
            res = max(res, disc)
            d[s[end]] = end
            
        return res
```    

<span id="jump49"></span>

49. [丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)    

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp, a, b, c = [1] * n, 0, 0, 0
        for i in range(1, n):
            # 动态规划，第i个丑数是第[1,i-1]个丑数*2，*3，*5的最小值
            # 设置a,b,c遍历整个[1,i-1]，dp[i] = min(dp[a]*2,dp[b]*3,dp[c]*5)
            n2, n3, n5 = dp[a] * 2, dp[b] * 3, dp[c] * 5
            dp[i] = min(n2, n3, n5)
            if dp[i] == n2: a += 1
            if dp[i] == n3: b += 1
            if dp[i] == n5: c += 1
        return dp[-1]
```    

<span id="jump50"></span>

50. [第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)   

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        d = {}
        for x in s:
            # 精妙所在
            # 相比 d[x]= d.get(x,0)+1
            d[x] = not x in d
        for k,v in d.items():
            if v:
                return k
        return " "
```    

<span id="jump51"></span>

51. [数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```python
class Solution:
    def mergeSort(self, nums, tmp, l, r):
        if l >= r:
            return 0
        mid = (l + r) // 2
        # 归并排序，不断切分数组
        cnt = self.mergeSort(nums, tmp, l, mid) + self.mergeSort(nums, tmp, mid+1, r)
        i, j, pos = l, mid+1, l
        # 二路归并
        while i <= mid or j <= r:
            if i > mid:
                tmp[pos]= nums[j]
                j += 1
            elif j > r:
                tmp[pos] = nums[i]
                i += 1
                # 每放入一个i计算有多少个j已经被放进去了。
                # 被放进去的j的个数等于逆序数
                cnt += (j-(mid+1))
            elif nums[i] <= nums[j]:
                tmp[pos] = nums[i]
                i += 1
                cnt += (j - (mid + 1))
            else:
                tmp[pos] = nums[j]
                j += 1
            pos += 1

        nums[l:r+1] = tmp[l:r+1]
        return cnt

    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)
        tmp = [0] * n
        return self.mergeSort(nums, tmp, 0, n-1)
```    

<span id="jump52"></span>

52. [两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        rootA, rootB = headA, headB
        """
        # 在后面添加来一个空指针，针对没有交点的特殊情况。
        # rootA和rootB都遍历A+B指针,消除A，B链表长度差。
        while rootA != rootB:
            rootA = rootA.next if rootA else headB
            rootB = rootB.next if rootB else headA
        return rootA
        """
        # 自己版本
        while rootA and rootB:
            rootA, rootB = rootA.next, rootB.next
        while rootA:
            rootA, headA = rootA.next, headA.next
        while rootB:
            rootB, headB = rootB.next, headB.next
        while headA and headB:
            if headA == headB:
                return headA
            headA, headB = headA.next, headB.next
        return
```    

<span id="jump53"></span>

53. [在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        res = 0
        for x in nums:
            if x == target:
                res += 1
        return res
```    

<span id="jump54"></span>

54. [0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        """
        # 直接遍历，复杂度n
        for i in range(len(nums)):
            if i != nums[i]:
                return i
        return i+1
        """
        # 二分法，复杂度logn
        # 寻找右数组的第一位，即下标不等于数值的第一位
        i, j = 0, len(nums) - 1
        # 优先判断首位和末位，如果满足条件则直接return，不二分
        if j == 0:
            return 0 if nums[0] == 1 else 1
        if nums[-1] != j+1:
            return j+1 
        while i < j:
            m = (i + j) // 2
            # 右数组的首位一定在m+1到j中
            if nums[m] == m:
                i = m + 1
            # 左数组的末位一定在l到m中
            else:
                j = m
        return nums[i] - 1
```    

<span id="jump55"></span>

55. [二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        # 逆中序遍历得到递减数组 
        stack = []
        while root or stack:
            if root:
                stack.append(root)
                root = root.right
            else:
                root = stack.pop()
                k -= 1
                if k == 0:
                    return root.val
                root = root.left
```    

<span id="jump56"></span>

56. [二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # 递归版
        # 递归子问题:maxDepth(root) = max(maxDepth(root.left) + maxDepth(root.right)) + 1

        if not root:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right)) + 1
```    

<span id="jump57"></span>

57. [平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # 用-1代表该节点的树深无限大
        def helper(root):
            if not root:
                return 0
            #左子树高
            left = helper(root.left)
            if left == -1:
                return -1
            #右子树高
            right = helper(root.right)
            if right == -1:
                return -1
            if abs(right - left) <= 1:
                return max(left, right) + 1
            else:
                return -1
        return helper(root) != -1
```    

<span id="jump58"></span>

58. [数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        """
        # 排序后查找，时间复杂度为O(NlogN)
        nums.sort()
        res = []
        i, l = 0, len(nums)
        while i < l - 1:
            if nums[i] == nums[i+1]:
                i += 2
            else:
                res.append(nums[i])
                i += 1
        if nums[l-1] != nums[l-2]:
            res.append(nums[l-1])
        return res
        """
        # 分组异或
        # 计算全员异或的值=a^b
        ret = functools.reduce(lambda x, y: x ^ y, nums)
        div = 1
        # 从右往左找到第一个不为0的数
        while div & ret == 0:
            div <<= 1
        a, b = 0, 0
        # 根据div进行分组异或
        for n in nums:
            if n & div:
                a ^= n
            else:
                b ^= n
        return [a, b]
```    

<span id="jump59"></span>

59. [数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        # 排序求解
        l = len(nums)
        if l == 1:
            return nums[0]
        nums.sort()
        i = 0
        while i < l-1:
            if nums[i] == nums[i+1]:
                i += 3
            else:
                return nums[i]
        return nums[-1]
        """

        # 通解，求重复出现m次数组里只出现一次的数字、
        # 位运算
        counts = [0] * 32
        for num in nums:
            for j in range(32):
                counts[j] += num & 1
                num >>= 1
                if num == 0:
                    break
        res, m = 0, 3
        for i in range(32):
            res <<= 1
            res |= counts[31 - i] % m
        return res if counts[31] % m == 0 else ~(res ^ 0xffffffff)
```    

<span id="jump60"></span>

60. [和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        二分查找+双指针
        """

        # 双指针查找
        def search(lo, hi):
            i, j = lo, hi
            while i < j:
                s = nums[i] + nums[j]
                if s > target: j -= 1
                elif s < target: i += 1
                else: return nums[i], nums[j]
            return []
        
        lo, hi = 0, len(nums)-1
        if hi == 0:
            return []
        # 二分查找，寻找目标区间
        while lo < hi:
            mid = (lo+hi) // 2
            if nums[mid] >= target:
                hi = mid - 1
            else:
                return search(lo, hi)
```    

<span id="jump61"></span>

61. [和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        res = []
        start, end,s = 1, 1, 0
        flag = target if target < 3 else (target // 2 + 2)
        while end <= flag:
            if s <= target:
                if s == target:
                    res.append([v for v in range(start,end)])
                s += end
                end += 1
            else:
                s -= start
                start += 1
        return res
```    

<span id="jump62"></span>

62. [翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        a = s.split(" ")
        a1 = [x for x in a if x != ""]
        return " ".join(a1[::-1])
```    

<span id="jump63"></span>

63. [左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n] + s[n+1:] + s[:n]
```

<span id="jump64"></span>

64. [滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or k == 0: return []
        deque = collections.deque()
        for i in range(k): # 未形成窗口   
            # 清除队列里小于押入值的元素  
            while deque and deque[-1] < nums[i]: deque.pop()
            deque.append(nums[i])
        res = [deque[0]]
        for i in range(k, len(nums)): # 形成窗口后  
            # 比较元素的大小，如果等于队列首位，则出队。 
            if deque[0] == nums[i - k]: deque.popleft()
            while deque and deque[-1] < nums[i]: deque.pop()
            deque.append(nums[i])
            res.append(deque[0])
        return res
```   

<span id="jump65"></span>

65. [队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

```python
class MaxQueue:

    def __init__(self):
        self.queue = []
        self.maxq = []


    def max_value(self) -> int:
        if not self.maxq:
            return -1
        return self.maxq[0]

    def push_back(self, value: int) -> None:
        self.queue.append(value)

        while self.maxq and self.maxq[-1] < value:
            self.maxq.pop()
        self.maxq.append(value)

    def pop_front(self) -> int:
        if not self.queue:
            return -1
        v = self.queue.pop(0)
        if v == self.maxq[0]:
            self.maxq.pop(0)
        return v


# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```   

<span id="jump66"></span>

66. [n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

```python
class Solution:
    def twoSum(self, n: int) -> List[float]:
        s = [1/6]*7
        d = { k:v for k,v in enumerate(s) if k > 0}
        while n > 1:
            res = {}
            for i in range(1,7):
                for k, v in d.items():
                    res[i+k] =res.get(i+k,0) + s[i] * v
            d = {k:v for k,v in res.items()}
            n -= 1
        return(d.values())
```   

<span id="jump67"></span>

67. [扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        # 本质：牌组中的最大值减去最小值小于5。  
        # 且牌组中除0以外的数字不能重复。  
        repeat = set()
        ma, mi = 0, 14
        for num in nums:
            if num == 0: continue # 跳过大小王
            ma = max(ma, num) # 最大牌
            mi = min(mi, num) # 最小牌
            if num in repeat: return False # 若有重复，提前返回 false
            repeat.add(num) # 添加牌至 Set
        return ma - mi < 5 # 最大牌 - 最小牌 < 5 则可构成顺子
```    

<span id="jump68"></span>

68. [圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/) 

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        if n == 1: return 0
        return (self.lastRemaining(n-1, m) + m) % n
```   

<span id="jump69"></span>

69. [股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 动态规划
        # dp[n]表示第0～n个元素中最大利润。    
        # 转移方程：dp[n] = max(dp[n-1], price[n]-min([:n]))。   
        # min([:n])可以在遍历的时候记录下来。   
        # 因为最后只取dp[n]，所以可以空间优化。   
        cost, profit = float("+inf"), 0
        for price in prices:
            cost = min(cost, price)
            profit = max(profit, price - cost)
        return profit
```

<span id="jump70"></span>

70. [求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

```python
class Solution:
    def __init__(self):
        self.res = 0
    def sumNums(self, n: int) -> int:
        n > 1 and self.sumNums(n-1)
        self.res += n
        return self.res
```
<span id="jump71"></span>

71. [不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

```python
class Solution:
    def add(self, a: int, b: int) -> int:
        # 加法等于异或加上进位与。
        # 此题还考察python的负数存储
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)
```

<span id="jump72"></span>

72. [构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        b = [1] * len(a)
        tmp = 1
        # 获取下标在自身之前的所有元素之积
        for i in range(1, len(a)):
            b[i] = b[i-1] * a[i-1]
        # 获取下标在自身之后的所有元素之积  
        for i in range(len(a)-2, -1, -1):
            tmp *= a[i+1]
            b[i] *= tmp
        return b
```

<span id="jump73"></span>

73. [把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

```python
class Solution:
    def strToInt(self, str: str) -> int:
        s = str.strip()
        r = []
        flag = ('+', '-')
        sign = ''


        for c in s:
            if c.isdigit(): 
                r.insert(0,c)
            elif c in flag and not r and not sign:
                sign = c
            else:
                break
        
        if not r:
            return 0
        
        x = 1
        res = 0
        for v in r:
            if v != sign:
                res += int(v) * x
                x *= 10

        res = res * -1 if sign == '-' else res * 1
        if res < -2 ** 31:
            res = - 2 ** 31
        if res >= 2 ** 31:
            res = 2 ** 31 - 1
        return res
```   

<span id="jump74"></span>

74. [二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return 
        if p == root or q == root:
            return root
        left = right = None
        if p.val < root.val or q.val < root.val:
            left = self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val or q.val > root.val:
            right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right
```

<span id="jump75"></span>

75. [二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root:
            return
        if p == root or q == root:
            return root
        left  = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right
```  



