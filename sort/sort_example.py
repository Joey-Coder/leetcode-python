import time
import numpy
import copy


class Sort:
    '各类排序集合类'

    def __init__(self, a: list):
        self.a = copy.copy(a)
        self.size = len(self.a)

    def less(self, v, w):
        return v < w

    def exch(self, i, j):
        self.a[i], self.a[j] = self.a[j], self.a[i]

    def show(self):
        print(self.a)

    def isSorted(self):
        for i in range(1, self.size):
            if self.less(self.a[i], self.a[i - 1]):
                return False
        return True

    def SelectSort(self):
        '选择排序'
        for i in range(self.size):
            Min = i
            for j in range(i + 1, self.size):
                if self.a[j] < self.a[Min]:
                    Min = j
            if Min != i:
                self.exch(Min, i)

    def InsertSort(self):
        '插入排序'
        for i in range(1, self.size):
            for j in range(i, 0, -1):
                if self.less(self.a[j], self.a[j - 1]):
                    self.exch(j, j - 1)
                else:
                    break

    def ShellSort(self):
        '希尔排序'
        '插入排序的升级版本'
        h = 1
        # 生成多个h
        while h < (self.size // 3):
            h = h * 3 + 1
        # h递减
        while h >= 1:
            # 从索引为h的元素一直到列表末尾进行插入排序，插入间隙为h
            for i in range(h, self.size):
                # j >= h而不是j > 0,两者都可以，但是使用j >= h可以使算法性能得到提升
                for j in range(i, h - 1, -h):
                    if self.less(self.a[j], self.a[j - h]):
                        self.exch(j, j - h)
                    else:
                        break
            # 进行h递减
            h = h // 3

    def MergeSort(self):
        # 辅助数组
        aux = [0] * self.size

        def merge(lo, mid, hi):
            i, j = lo, mid + 1
            for k in range(lo, hi + 1):
                aux[k] = self.a[k]

            for k in range(lo, hi + 1):
                if i > mid:
                    self.a[k] = aux[j]
                    j += 1
                elif j > hi:
                    self.a[k] = aux[i]
                    i += 1
                elif aux[j] < aux[i]:
                    self.a[k] = aux[j]
                    j += 1
                else:
                    self.a[k] = aux[i]
                    i += 1

        # 子数组大小
        sz = 1
        while sz < self.size:
            # 从左到右进行遍历，以sz*2大小进行归并
            for lo in range(0, self.size - sz, sz + sz):
                merge(lo, lo + sz - 1,
                      min(lo + sz + sz - 1, self.size - 1))
            # 子数组大小变为原来的2倍
            sz = sz + sz

    def QuickSort(self, lo=0, hi=None):
        def partition(lo, hi):
            i, j = lo, hi + 1
            v = self.a[lo]
            while True:
                while True:
                    i += 1
                    if i == hi or self.a[i] >= v:
                        break
                while True:
                    j -= 1
                    if j == lo or self.a[j] <= v:
                        break
                if i >= j:
                    break
                self.exch(i, j)
            self.exch(lo, j)
            return j

        def insertsort(lo, hi):
            for i in range(lo + 1, hi + 1):
                for j in range(i, 0, -1):
                    if self.less(self.a[j], self.a[j - 1]):
                        self.exch(j, j - 1)
                    else:
                        break

        if hi is None:
            hi = self.size - 1
        if lo < hi:
            if hi - lo <= 10:
                # 当递归的数组大小小于10的时候，直接使用插入排序
                # 插入排序在5-15的数量级效果会比快排好
                insertsort(lo, hi)
            else:
                j = partition(lo, hi)
                self.QuickSort(lo, j - 1)
                self.QuickSort(j + 1, hi)

    # 快速排序改进版，使用三向切分，针对重复元素较多时使用
    # 如果没有大量的重复元素的话，其效率比普通的快速排序低
    def ThreeQuickSort(self, lo=0, hi=None):
        def insertsort(lo, hi):
            for i in range(lo + 1, hi + 1):
                for j in range(i, 0, -1):
                    if self.less(self.a[j], self.a[j - 1]):
                        self.exch(j, j - 1)
                    else:
                        break

        if hi is None:
            hi = self.size - 1
        if lo < hi:
            if lo >= hi - 10:
                insertsort(lo, hi)
            else:
                lt, i, gt = lo, lo + 1, hi
                v = self.a[lo]
                while i <= gt:
                    t = self.a[i] - v
                    if t < 0:
                        self.exch(lt, i)
                        lt += 1
                        i += 1
                    elif t > 0:
                        self.exch(i, gt)
                        gt -= 1
                    else:
                        i += 1
                self.ThreeQuickSort(lo, lt - 1)
                self.ThreeQuickSort(gt + 1, hi)

    def HeapSort(self):
        # 下沉函数，判断父节点是否小于小个子节点
        # 如果小于，则将父节点和子节点中较大的一个对换
        def sink(k, N):
            while 2 * k <= N:
                j = 2 * k
                # j < N是为了区分只有一个子节点或者没有子节点的情况
                if j < N and self.less(self.a[j], self.a[j + 1]):
                    j += 1
                if not self.less(self.a[k], self.a[j]):
                    break
                self.exch(k, j)
                k = j

        # 堆的第一个元素不存储，所以这里往0位置插入None
        N = self.size
        self.a.insert(0, -1)
        # 第一次堆有序
        for k in range(N // 2, 0, -1):
            sink(k, N)
        while N > 1:
            # 将堆首最大元素移到最后一位
            self.exch(1, N)
            # 缩小堆容量，进行堆有序操作
            N -= 1
            sink(1, N)

    def PythonSort(self):
        self.a = sorted(self.a)

    def costTime(self, sortname: str):
        sortname = sortname.lower()
        start = time.time()
        if sortname == 'selectsort':
            self.SelectSort()
        elif sortname == "insertsort":
            self.InsertSort()
        elif sortname == "shellsort":
            self.ShellSort()
        elif sortname == "mergesort":
            self.MergeSort()
        elif sortname == "quicksort":
            self.QuickSort()
        elif sortname == "pythonsort":
            self.PythonSort()
        elif sortname == "threequicksort":
            self.ThreeQuickSort()
        elif sortname == "heapsort":
            self.HeapSort()
        end = time.time()
        if self.isSorted():
            print("{0} cost time {1}".format(sortname, end - start))
        else:
            print("sort failed!")


if __name__ == '__main__':
    n = input("Please input the list size: ")
    a = list(numpy.random.randint(100000, size=int(n)))
    # a = [23] * int(n)
    """
    s = Sort(a)
    s.costTime("SelectSort")
    s = Sort(a)
    s.costTime("InsertSort")
    s = Sort(a)
    s.costTime("ShellSort")
    s = Sort(a)
    s.costTime("ThreeQuickSort")
    """
    s = Sort(a)
    s.costTime("MergeSort")
    s = Sort(a)
    s.costTime("QuickSort")
    s = Sort(a)
    s.costTime("PythonSort")
    s = Sort(a)
    s.costTime("HeapSort")
