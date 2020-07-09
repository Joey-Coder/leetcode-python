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