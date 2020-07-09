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
        root = helper(preorder, 0, len(preorder) - 1, inorder, 0, len(inorder) - 1)
        return root