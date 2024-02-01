# 1. Two Sum (Easy)
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, j in enumerate(nums):
            if i==0:
                d[j] = i
            elif target-j in d.keys():
                return [d[target-j], i]
            else:
                d[j] = i

# 2. Add Two Numbers (Medium)
# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
# You may assume the two numbers do not contain any leading zero, except the number 0 itself.

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        curr = ListNode(0)
        headnode = curr
        while l1 != None or l2!= None:
            l1v = l1.val if l1!= None else 0
            l2v = l2.val if l2!= None else 0
            curr.val += l1v + l2v 
            curr1 = ListNode(curr.val // 10)
            curr.val = curr.val % 10 
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            if l1 or l2 or curr1.val!=0:
                curr.next = curr1
                curr = curr.next
        return headnode

# 3. Longest Substring Without Repeating Characters (Medium)
# Given a string s, find the length of the longest substring without repeating characters.

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = 0
        m = 0
        d = []
        for i in s:
            if i not in d: 
                d.append(i)
                n += 1
                # print('adding ', i, d, n, m)
            else:
                # print(i, d.index(i), d, n, m)
                d = d[d.index(i)+1:]
                d.append(i)
                if n > m:
                    m = n
                n = len(d)
                # print('now: ', d, n, m)
        if n > m:
            m = n
        # print(d, n)
        return m

# 4. Median of Two Sorted Arrays (Hard)
# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).


# 9. Palindrome Number
# Given an integer x, return true if x is a palindrome, and false otherwise.
    
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x<0:
            return False
        n = 0
        xc = x
        while xc!=0:
            n = n*10+xc%10
            xc = xc // 10
        if n == x:
            return True
        return False


# 13. Roman to Integer
# Given a roman numeral, convert it to an integer.

class Solution:
    def romanToInt(self, s: str) -> int:
        rim = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        n = 0
        i = 0
        for i in range(1, len(s)):
            if rim[s[i]]>rim[s[i-1]]:
                n -= rim[s[i-1]]
            else:
                n += rim[s[i-1]]
        n += rim[s[i]]
        return n

# 14. Longest Common Prefix
# Write a function to find the longest common prefix string amongst an array of strings.

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        pref = ''
        for j in range(len(strs[0])):
            for i in range(len(strs)):
                if len(strs[i])<=j:
                    return pref
                if strs[i][j]!=strs[0][j]:
                    return pref
            pref += strs[0][j]
        return pref
            

# 20. Valid Parentheses
# Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

class Solution:
    def isValid(self, s: str) -> bool:
        opened = []
        for i in s:
            if i in ['(', '{', '[']:
                opened.append(i)
            if i in [')', '}', ']']:
                if len(opened)>=1:
                    if i==')' and opened[-1]=='(':
                        opened.pop()
                    elif i=='}' and opened[-1]=='{':
                        opened.pop()
                    elif i==']' and opened[-1]=='[':
                        opened.pop()
                    else:
                        return False
                else:
                    return False
        if len(opened)>=1:
            return False
        return True



# 21. Merge Two Sorted Lists
# You are given the heads of two sorted linked lists list1 and list2.
# Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
# Return the head of the merged linked list.

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode()
        curr = head
        while list1 and list2:
            if list1.val<=list2.val:
                curr.next = list1
                curr = list1 
                list1 = list1.next
            else:
                curr.next = list2
                curr = list2
                list2 = list2.next
        if list1 or list2:
            curr.next = list1 if list1 else list2 
        return head.next

# 26. Remove Duplicates from Sorted Array
# Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same. Then return the number of unique elements in nums.

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = 1
        l = nums[-1]
        i = 1
        if len(nums)==1:
            return 1
        while nums[i]!=l:
            if nums[i]!=nums[i-1]:
                nums[n] = nums[i]
                n += 1
            i += 1
        if nums[i]!=nums[i-1]:
            nums[n] = nums[i]
            n += 1
        return n


# 27. Remove Element
# Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        n = 0
        for i in range(len(nums)):
            if nums[i]!=val:
                nums[n] = nums[i]
                n += 1
        return n

# 35. Search Insert Position
# Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        for i, j in enumerate(nums):
            if target<=j:
                return i
        return len(nums)

# 70. Climbing Stairs
# You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

def factorial(n):
    s = 1
    for i in range(1, n+1):
        s *= i
    return s

def sochetanie(a, b):
    return factorial(b) / factorial(b - a) / factorial(a)

class Solution(object):
    def climbStairs(self, n):
        if n==1:
            return 1
        s = 0 
        for i in range(n//2+1):
            s += sochetanie(i, i+n-i*2)
        return int(s)
        
# 121. Best Time to Buy and Sell Stock
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

class Solution(object):
    def maxProfit(self, prices):
        min_price = prices[0]
        max_profit = 0

        for i in range(1, len(prices)):
            price = prices[i]
            max_profit = max(price - min_price, max_profit)
            min_price = min(price, min_price)

        return max_profit



# 101. Symmetric Tree
# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        left_side = root
        right_side = root

        def check_sides(left_side, right_side):
            if left_side.val != right_side.val:
                return False
            if left_side.left and right_side.right:
                f = check_sides(left_side.left, right_side.right)
                if f == 0:
                    return False
            elif left_side.left or right_side.right: 
                return False
            if left_side.right and right_side.left:
                f = check_sides(left_side.right, right_side.left)
                if f == 0:
                    return False
            elif left_side.right or right_side.left: 
                return False
            return True
        
        return check_sides(left_side, right_side)


# 104. Maximum Depth of Binary Tree
# Given the root of a binary tree, return its maximum depth.
# A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        if not root:
            return 0 
        def check_depth(curr):
            if curr.left or curr.right:
                if curr.left:
                    leftl= check_depth(curr.left)
                else:
                    leftl= 0
                if curr.right:
                    rightl= check_depth(curr.right) 
                else:
                    rightl= 0
                if leftl>=rightl:
                    return 1 + leftl
                else:
                    return 1 + rightl
            return 1
        
        return check_depth(root)

                

# 5. Longest Palindromic Substring
# Given a string s, return the longest palindromic substring in s.

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        for maxl in range(n, 1, -1):
            for j in range(0, n-maxl+1):
                x = 0
                while s[j+x]==s[maxl+j-x-1] and x<maxl//2:
                    x += 1
                if x>=maxl//2:
                    return s[j:j+maxl]
        return s[0]


# 11. Container With Most Water
# You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
# Find two lines that together with the x-axis form a container, such that the container contains the most water.
# Return the maximum amount of water a container can store.

class Solution:
    def maxArea(self, height) -> int:
        n = len(height)-1
        m = 0 
        max_pool = 0
        while m<n:
            if height[m]>height[n]:
                curr_pool = height[n]*(n-m)
                n -= 1
            else:
                curr_pool = height[m]*(n-m)
                m += 1
            if curr_pool>max_pool:
                max_pool = curr_pool

        return max_pool


# 15. 3Sum
# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# Notice that the solution set must not contain duplicate triplets.

def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

def quickSort(array, low, high):
	if low < high:
		pi = partition(array, low, high)
		quickSort(array, low, pi - 1)
		quickSort(array, pi + 1, high)

class Solution:
    def threeSum(self, nums):
        result = []
        n = len(nums)
        quickSort(nums, 0, n-1)
        if nums[0]>0 or nums[n-1]<0:
            return result
        i = 0
        j = n - 1
        while nums[i]<=0 and i<n-2:
            curr = nums[i]
            while nums[j]>-nums[i]*2 and j>i:
                j -= 1
            left = i+1
            right = j
            while left<right and nums[right]>=0:
                if nums[left]+nums[right]==-nums[i]:
                    if len(result)==0:
                        result.append([nums[i], nums[left], nums[right]])
                    elif nums[i]!=result[-1][0] or nums[right]!=result[-1][2]:
                        result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                elif nums[left]+nums[right]>-nums[i]:
                    right -=1
                else:
                    left +=1
            i += 1
            while nums[i]==curr and i<n-1:
                i += 1
        return result

# 17. Letter Combinations of a Phone Number
# Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
# A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

def add_letter(result, letters):
    new_result = []
    for i in result:
        for j in letters:
            new_result.append(i+j)
    if result==[]:
        for j in letters:
            new_result.append(j)
    return new_result

class Solution:
    def letterCombinations(self, digits: str):
        phone = {'2':['a', 'b', 'c'], '3':['d', 'e', 'f'], 
                '4':['g', 'h', 'i'], '5':['j', 'k', 'l'], 
                '6':['m', 'n', 'o'], '7':['p', 'q', 'r', 's'], 
                '8':['t', 'u', 'v'], '9':['w', 'x', 'y', 'z']}

        s = []
        for i in digits:
            s = add_letter(s, phone[i])
        return s

# 141. Linked List Cycle
# Given head, the head of a linked list, determine if the linked list has a cycle in it.
# There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
# Return true if there is a cycle in the linked list. Otherwise, return false.

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        run2 = head
        while head.next and run2.next and run2.next.next:
            head = head.next
            run2 = run2.next.next
            if head==run2:
                return True
        return False

# 22. Generate Parentheses
# Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []

        def add_bracket(strn, fo, fc):
            if fo==n and fc==n:
                result.append(strn)
                return 
            if fo<n:
                add_bracket(strn+'(', fo+1, fc)
            if fc<n and fc<fo:
                add_bracket(strn+')', fo, fc+1)
        
        add_bracket('(', 1, 0)
        return result


# 169. Majority Element
# Given an array nums of size n, return the majority element.
# The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n = 0
        value = 0
        for i in range(len(nums)):
            if n == 0:
                value = nums[i]
            if nums[i]==value:
                n += 1
            else:
                n -= 1
        return value


# 206. Reverse Linked List
# Given the head of a singly linked list, reverse the list, and return the reversed list.

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        root = None
        while head:
            head.next, root, head = root, head, head.next
        
        return root

# 226. Invert Binary Tree
# Given the root of a binary tree, invert the tree, and return its root.

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return 
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root 

# 234. Palindrome Linked List
# Given the head of a singly linked list, return true if it is a palindrome or false otherwise.

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        root = ListNode()
        head_head = head
        head_root = root
        while head:
            root.val = head.val
            root.next = ListNode()
            head = head.next
            root = root.next

        end = None
        while head_head:
            head_head.next, end, head_head = end, head_head, head_head.next

        while head_root and end:
            if head_root.val!=end.val:
                return False
            head_root = head_root.next
            end = end.next
        return True

# 160. Intersection of Two Linked Lists
# Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode):
        headAc = headA
        headBc = headB
        headAcc = headA
        headBcc = headB
        while headAcc and headAcc.next:
            headAcc = headAcc.next
        while headBcc and headBcc.next:
            headBcc = headBcc.next
        if headBcc!=headAcc:
            return
        

        while headA!=headB:
            if headA:
                headA = headA.next
            else:
                headA = headAc
            if headB:
                headB = headB.next
            else:
                headB = headBc
        return headA
        

# 283. Move Zeroes
# Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
# Note that you must do this in-place without making a copy of the array.
        
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        z = -1
        for i in range(len(nums)):
            if nums[i]==0 and z==-1:
                z = i
            if nums[i]!=0 and z!=-1:
                nums[z] = nums[i]
                nums[i] = 0
                z += 1

# 704. Binary Search
# Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.
# You must write an algorithm with O(log n) runtime complexity.

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0 
        r = len(nums)-1
        while l<r-1:
            m = l+(r-l)//2
            if nums[m]==target:
                return m
            elif nums[m]>target:
                r = m 
            else:
                l = m
        if nums[l]==target:
            return l
        if nums[r]==target:
            return r
        return -1

# 118. Pascal's Triangle
# Given an integer numRows, return the first numRows of Pascal's triangle.

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        result = [[1]]

        def getcalc(result):
            result1 = [1]
            for i in range(1, len(result)):
                result1.append(result[i]+result[i-1])
            result1.append(1)
            return result1

        while numRows>1:
            numRows -= 1
            result.append(getcalc(result[-1]))
        return result

# 543. Diameter of Binary Tree
# Given the root of a binary tree, return the length of the diameter of the tree.
# The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
# The length of a path between two nodes is represented by the number of edges between them.

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        result = []

        def depth(tree):
            if not tree:
                return -1
            l = depth(tree.left)
            r = depth(tree.right)
            result.append(2+l+r)
            return 1+max(l, r)

        depth(root)
        return max(result)

# 24. Swap Nodes in Pairs
# Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        n = 0
        if head and head.next:
            headc = head.next
        else:
            headc = head
        curr = ListNode()
        while head and head.next:
            tmp = head.next
            head.next = head.next.next
            tmp.next = head
            curr.next = tmp
            curr = head
            head = head.next
        return headc







