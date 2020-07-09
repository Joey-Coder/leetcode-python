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