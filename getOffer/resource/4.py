class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        # 利用栈的FILO
        stack = []
        while head:
            stack.insert(0, head.val)
            head = head.next

        return stack