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