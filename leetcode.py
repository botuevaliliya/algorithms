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


# 31. Next Permutation
# A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

def partition(array, low, high, asc):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot and asc==1:
            i = i + 1
            array[i], array[j] = array[j], array[i]
        if array[j] >= pivot and asc==0:
            i = i + 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

def quickSort(array, low, high, asc):
	if low < high:
		pi = partition(array, low, high, asc)
		quickSort(array, low, pi - 1, asc)
		quickSort(array, pi + 1, high, asc)

class Solution:
    def nextPermutation(self, nums) -> None:
        l = len(nums)
        n = l-1
        while n>0 and nums[n-1]>=nums[n]:
            n -= 1
        if n==0:
            quickSort(nums, 0, l-1, 1)
            return
        m = 101
        mi = l
        for i in range(n, l):
            if nums[i]<m and nums[i]>nums[n-1]:
                m = nums[i]
                mi = i
        if mi!=l:
            nums[n-1], nums[mi] = m, nums[n-1]
            quickSort(nums, n, l-1, 1)
            return
        quickSort(nums, n-1, l-1, 0)

# 33. Search in Rotated Sorted Array
# There is an integer array nums sorted in ascending order (with distinct values).
# Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
# Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

class Solution:
    def search(self, nums, target: int) -> int:
        l = 0 
        r = len(nums)-1
        if nums[r]==target:
            return r
        elif nums[l]==target:
            return l
        while l<r-1:
            m = l+(r-l)//2
            if nums[m]==target:
                return m
            elif nums[r]==target:
                return r
            elif nums[l]==target:
                return l
            elif l==m or m==r:
                return -1
            elif nums[m]>target and target>nums[l]:
                r = m 
            elif nums[m]<target and target<nums[r]:
                l = m
            elif target>nums[m] and target>nums[r] and nums[r]>nums[m]:
                r = m
            elif target>nums[m] and target>nums[r] and nums[r]<nums[m]:
                l = m
            elif target<nums[m] and nums[r]<nums[m]:
                l = m
            elif target<nums[m] and nums[r]>nums[m]:
                r = m
            else:
                return -1
        return -1

# 34. Find First and Last Position of Element in Sorted Array
# Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.
# If target is not found in the array, return [-1, -1].
# You must write an algorithm with O(log n) runtime complexity.

class Solution:
    def searchRange(self, nums, target: int):
        l = 0 
        r = len(nums)-1
        if nums==[]:
            return [-1, -1]
        m = 0
        if nums[l]==target:
            m=l
        if nums[r]==target:
            m=r
        while l<r-1:
            m = l+(r-l)//2
            if nums[m]==target:
                break 
            elif nums[l]==target:
                m=l
                break
            elif nums[r]==target:
                m=r
                break
            elif nums[m]>target:
                r = m 
            else:
                l = m
        k = m
        if nums[m]==target:
            while k>=0 and nums[k]==target:
                k -= 1
            while m<=r and nums[m]==target:
                m += 1
            return [k+1, m-1]
        return [-1, -1]


# 39. Combination Sum
# Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.
# The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.
# The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

class Solution:
    def combinationSum(self, candidates, target: int):
        result = []

        def all_solutions(comb, s):
            if s>target:
                return 
            if s==target:
                for res in result:
                    if res==comb:
                        return
                result.append(comb)
                return 
            for i in candidates:
                if comb==[]:
                    tmp = comb + [i]
                    all_solutions(tmp, s+i)
                elif i>=comb[-1]:
                    tmp = comb + [i]
                    all_solutions(tmp, s+i)
        
        all_solutions([], 0)
        return result

# 45. Jump Game II
# You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
# Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where: 0 <= j <= nums[i] and i + j < n
# Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].

class Solution:
    def jump(self, nums) -> int:
        n = len(nums) - 1
        i = n
        mi = n
        qty = 0
        while i>0:
            k = mi-1
            m = mi-1
            while k>=0:
                if mi-k<=nums[k]:
                    m = k
                k -= 1
            i = m
            mi = i
            qty += 1
        return qty

# 46. Permutations
# Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        n = len(nums)

        def all_solutions(comb, s):
            if s==n:
                result.append(comb)
                return 
            for i in nums:
                if comb==[]:
                    tmp = comb + [i]
                    all_solutions(tmp, s+1)
                elif i not in comb:
                    tmp = comb + [i]
                    all_solutions(tmp, s+1)
        
        all_solutions([], 0)
        return result

# 48. Rotate Image
# You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
# You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

class Solution:
    def rotate(self, matrix):
        n = len(matrix[0])
        nn = n
        i = 0
        j = 0
        while n>=2:
            for j in range(i, n-1):
                matrix[i][j], matrix[j][nn-i-1], matrix[nn-i-1][nn-j-1], matrix[nn-j-1][i] = matrix[nn-j-1][i], matrix[i][j], matrix[j][nn-i-1], matrix[nn-i-1][nn-j-1]
            n -= 1
            i += 1

# 49. Group Anagrams
# Given an array of strings strs, group the anagrams together. You can return the answer in any order.
# An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

class Solution:
    def groupAnagrams(self, strs):
        has = {}
        for i in strs:
            k = ''.join(sorted(i))
            if k not in has.keys():
                has[k] = [i]
            else:
                has[k].append(i)
        return list(has.values())

# 54. Spiral Matrix
# Given an m x n matrix, return all elements of the matrix in spiral order.

class Solution:
    def spiralOrder(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        result = []
        i = 0
        j = 0
        s = n*m

        def add_line(i, j, point, s, n, m):
            if s==0:
                return 
            if point=='r':
                for incr in range(n):
                    result.append(matrix[i][j+incr])
                s -= n
                m -= 1
                add_line(i+1, j+incr, 'd', s, n, m)
            if point=='d':
                for incr in range(m):
                    result.append(matrix[i+incr][j])
                s -= m
                n -= 1
                add_line(i+incr, j-1, 'l', s, n, m)
            if point=='l':
                for incr in range(n):
                    result.append(matrix[i][j-incr])
                s -= n
                m -= 1
                add_line(i-1, j-incr, 'u', s, n, m)
            if point=='u':
                for incr in range(m):
                    result.append(matrix[i-incr][j])
                s -= m
                n -= 1
                add_line(i-incr, j+1, 'r', s, n, m)
            
        add_line(0, 0, 'r', s, n, m)
        return result

# 55. Jump Game
# You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
# Return true if you can reach the last index, or false otherwise.

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums) - 1
        mi = n
        for i in range(n,-1,-1):
            if mi-i<=nums[i]:
                    mi = i
        if mi==0:
            return True
        return False

# 56. Merge Intervals
# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

class Solution:
    def merge(self, intervals):
        n = len(intervals)
        intervals = sorted(intervals)
        result = [intervals[0]]
        for i in range(1, n):
            if intervals[i][0]<=result[-1][1]:
                if result[-1][1]<intervals[i][1]:
                    result[-1][1] = intervals[i][1]
            else:
                result.append(intervals[i])
                
        return result
            
# 62. Unique Paths
# There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
# The test cases are generated so that the answer will be less than or equal to 2 * 109.

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # def all_solutions(m, n):
        #     if m==1 or n==1:
        #         return 1
        #     return all_solutions(m-1, n) + all_solutions(m, n-1)

        # return all_solutions(m, n)
        result = [[]]
        for i in range(m):
            for j in range(n):
                if i==0 or j==0:
                    result[i].append(1)
                else:
                    result[i].append(result[i-1][j] + result[i][j-1])
            result.append([])
        return result[m-1][n-1]

# 64. Minimum Path Sum
# Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
# Note: You can only move either down or right at any point in time.

class Solution:
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])
        result = []
        for i in range(m):
            result.append([])
            for j in range(n):
                if i==0 and j==0:
                    result[i].append(grid[i][j])
                elif i==0:
                    result[i].append(grid[i][j]+result[i][j-1])
                elif j==0:
                    result[i].append(grid[i][j]+result[i-1][j])
                else:
                    result[i].append(min(result[i-1][j]+grid[i][j], grid[i][j] + result[i][j-1]))
        return result[m-1][n-1]

# 73. Set Matrix Zeroes
# Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
# You must do it in place.

class Solution:
    def setZeroes(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        zrr = []
        zrc = []
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    zrr.append(i)
                    zrc.append(j)
        for i in zrr:
            for j in range(n):
                matrix[i][j]=0
        for j in zrc:
            for i in range(m):
                matrix[i][j]=0
                
# 74. Search a 2D Matrix
# You are given an m x n integer matrix matrix with the following two properties:
# Each row is sorted in non-decreasing order.
# The first integer of each row is greater than the last integer of the previous row.
# Given an integer target, return true if target is in matrix or false otherwise.
# You must write a solution in O(log(m * n)) time complexity.

class Solution:
    def searchMatrix(self, matrix, target):
        left = 0 
        right = len(matrix)-1
        while left<right-1:
            middle = left+(right-left)//2
            if matrix[middle][-1]==target:
                left = middle
                break
            elif matrix[middle][-1]>target:
                right = middle 
            else:
                left = middle
        if matrix[left][-1]<target:
            left = right 
        l = 0 
        right = len(matrix[0])-1
        while l<right-1:
            middle = l+(right-l)//2
            if matrix[left][middle]==target:
                l = middle
                break
            elif matrix[left][middle]>target:
                right = middle 
            else:
                l = middle
        if matrix[left][right]==target or matrix[left][l]==target:
            return True
        return False


# 75. Sort Colors
# Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
# We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
# You must solve this problem without using the library's sort function.

class Solution:
    def sortColors(self, nums):
        l = 0
        r = len(nums) - 1
        fl = 0
        fr = 0
        while l<r and (fl!=-1 or fr!=-1):
            mn = l
            while mn<=r and nums[mn]!=0:
                mn += 1
            if mn == r+1:
                fl = -1
            elif fl!= -1:
                nums[mn], nums[l] = nums[l], nums[mn]
                l += 1
            mx = r 
            while mx>=l and nums[mx]!=2:
                mx -= 1
            if mx == l-1:
                fr = -1
            elif fr!=-1:
                nums[mx], nums[r] = nums[r], nums[mx]
                r -= 1

# 78. Subsets
# Given an integer array nums of unique elements, return all possible subsets (the power set).
# The solution set must not contain duplicate subsets. Return the solution in any order.

class Solution:
    def subsets(self, nums):
        result = []
        n = len(nums)

        def all_subsets(ind, n, sub):
            if ind==n:
                result.append(sub)
                return
            all_subsets(ind+1, n, sub+[nums[ind]])
            all_subsets(ind+1, n, sub)

        all_subsets(0, n, [])
        return result

# 98. Validate Binary Search Tree
# Given the root of a binary tree, determine if it is a valid binary search tree (BST).
# A valid BST is defined as follows:
# The left subtree of a node contains only nodes with keys less than the node's key.
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.

class Solution:
    def isValidBST(self, root):
        
        def isValidBST1(root, mn, mx):
            if not root:
                return True
            if root.val<=mn or root.val>=mx:
                return False
            l = isValidBST1(root.left, mn, root.val)
            r = isValidBST1(root.right, root.val, mx)
            return r*l
        return isValidBST1(root, float('-inf'), float('inf'))
        
# 102. Binary Tree Level Order Traversal       
# Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        result = {0:[root.val]}

        def levelvals(root, level):
            if not root:
                return 
            l = levelvals(root.left, level+1)
            r = levelvals(root.right, level+1)
            if level not in result and (l!=None or r!=None):
                result[level] = []
            if l!=None: 
                result[level].append(l)
            if r!=None:
                result[level].append(r)

            return root.val

        levelvals(root, 1)
        print(result)
        return list(dict(sorted(result.items())).values())


# 128. Longest Consecutive Sequence
# Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
# You must write an algorithm that runs in O(n) time.

class Solution:
    def longestConsecutive(self, nums) -> int:
        nums.sort()
        j = 1
        m = 1
        if nums ==[]:
            return 0
        cm = nums[0]
        for i in range(1, len(nums)):
            if nums[i]==cm+1:
                j += 1
                cm = nums[i]
            elif nums[i]!=cm:
                if j>=m:
                    m = j
                j = 1
                cm = nums[i]
        if j>=m:
            m = j
        return m

# 142. Linked List Cycle II
# Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.
# There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. Note that pos is not passed as a parameter.
# Do not modify the linked list.

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return 
        has = {}
        while head not in has:
            if head.next==None:
                return 
            has[head] = 1
            head = head.next
        return head


# 114. Flatten Binary Tree to Linked List
# Given the root of a binary tree, flatten the tree into a "linked list":
# The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
# The "linked list" should be in the same order as a pre-order traversal of the binary tree.

class Solution:
    def flatten(self, root) -> None:

        def depth(tree):
            if not tree:
                return 
            
            l = depth(tree.left)
            r = depth(tree.right)

            if tree.left:
                l.right = tree.right
                tree.right = tree.left
                tree.left = None
            last = r or l or tree
            return last
            
        depth(root)

# 53. Maximum Subarray
# Given an integer array nums, find the subarray with the largest sum, and return its sum.

class Solution:
    def maxSubArray(self, nums):
        # result = []
        # n = len(nums)

        # def arraysum(nums, i, j):
        #     result.append(sum(nums[i:j+1]))
        #     if i+1<=j:
        #         arraysum(nums, i+1, j)
        #         arraysum(nums, i, j-1)
        #     return
            
        # arraysum(nums, 0, n-1)
        # return max(result)

        # local max->sum between local max->merge close local max if improve
        # local_max_or_pos = {}
        # between_locals = {}
        # n = len(nums)
        # if n==1:
        #     return nums[0]
        # last_key = -1
        # max_pos = -2
        # for i in range(n):
        #     if nums[i]>=0:
        #         if max_pos==i-1:
        #             local_max_or_pos[last_key] += nums[i]
        #             max_pos = i
        #         elif last_key!=i-1 or last_key==-1:
        #             local_max_or_pos[i] = nums[i]
        #             last_key = i
        #             max_pos = i
        #         else:
        #             local_max_or_pos[last_key] += nums[i]
        #             max_pos = i
        #     elif i==0 and nums[i]>=nums[i+1]:
        #         local_max_or_pos[i] = nums[i]
        #         last_key = i
        #     elif i==n-1 and nums[i]>=nums[i-1]:
        #         local_max_or_pos[i] = nums[i]
        #         last_key = i
        #     elif nums[i]>=nums[i-1] and nums[i]>=nums[i+1]:
        #         local_max_or_pos[i] = nums[i]
        #         last_key = i
        #     elif local_max_or_pos!={}:
        #         if last_key==i-1 or max_pos==i-1:
        #             between_locals[last_key] = nums[i]
        #         else:
        #             between_locals[last_key] += nums[i]
            
        # lm = list(local_max_or_pos.values())
        # if len(lm)==1:
        #     return lm[0]
        # bl = list(between_locals.values())
        # i = 1
        # while i<len(lm) and i<=len(bl):
        #     if lm[i-1]>-bl[i-1] and lm[i]>-bl[i-1]:
        #         lm[i] += lm[i-1]+bl[i-1]
        #     i += 1
        # return max(lm)

        s = nums[0]
        mx = s
        for i in range(1, len(nums)):
            s += nums[i]
            if s>=mx:
                mx = s
            if nums[i]>=mx:
                mx = nums[i]
                s = nums[i]
            elif nums[i]>=s:
                s = nums[i]
            
        return max(s, mx)
            

# 131. Palindrome Partitioning
# Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

class Solution:
    def partition(self, s):
        result = []

        def comb(lst, s, f):
            n = len(s)
            if n == 1:
                lst.append(s)
                # print('end')
                result.append(lst)
                return
            for i in range(1, n+1):
                if f==1:
                    lst = []
                x = i//2
                # print(s[:x], s[i-1:i-x-1:-1], s, x, i, s[:i], s[i:])
                if s[:x]==s[i-1:i-x-1:-1] or i==1:
                    # print('congrats')
                    if i==n:
                        result.append(lst + [s[:i]])
                        return
                    else:
                        # print('HERE', lst, s[:i])
                        comb(lst + [s[:i]], s[i:], 0)
                
        comb([], s, 1)
        
        return result

# 72. Edit Distance
# Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
# You have the following three operations permitted on a word:
# Insert a character
# Delete a character
# Replace a character

class Solution:
    def minDistance(self, word1, word2):
        n, m = len(word1), len(word2)
        if n > m:
            word1, word2 = word2, word1
            n, m = m, n
        current_row = list(range(n + 1))
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if word1[j - 1] != word2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        return current_row[n]
            
            
# 79. Word Search
# Given an m x n grid of characters board and a string word, return true if word exists in the grid.
# The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

class Solution:
    def exist(self, board, word):
        shape = [len(board), len(board[0])]
        word = list(word)
        l = len(word)

        def letter(i, j, k, l, mp):
            # print(i, j, mp, k, word[k])
            if k == l: 
                return True
            if i<0 or j<0 or i>=shape[0] or j>=shape[1] or k>=l:
                return False
            if board[i][j]!=word[k]:
                return False
            if [i,j] in mp:
                return False
            mp = mp[:k]
            mp.append([i,j])
            return letter(i+1, j, k+1, l, mp) or letter(i-1, j, k+1, l, mp) or letter(i, j+1, k+1, l, mp) or letter(i, j-1, k+1, l, mp)


        for i in range(shape[0]):
            for j in range(shape[1]):
                # mp = [[0] * shape[1] for p in range(shape[0])]
                if letter(i, j, 0, l, [])==True:
                    return True
        return False



# 105. Construct Binary Tree from Preorder and Inorder Traversal
# Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, preorder, inorder):
        l = len(preorder)
        root = TreeNode(preorder[0])
        parent = [0]
        
        def dfs(i, j, link):
            if j==i:
                return link
            k = i
            while k<=j and preorder[parent[0]]!=inorder[k]:
                k += 1
            if k>i:
                parent[0] += 1
                ln = dfs(i, k-1, TreeNode(preorder[parent[0]]))
                link.left = ln
            if k<j:
                parent[0] += 1
                rn = dfs(k+1, j, TreeNode(preorder[parent[0]]))
                link.right = rn
            return link

        dfs(0, l-1, root)
        return root 

# 138. Copy List with Random Pointer
# A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.
# Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.
# For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.
# Return the head of the copied linked list.

"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return

        save = head
        root = Node(head.val)
        save_new = root
        save1 = head
        save_new1 = root
        has = {}
        while head and head.next:
            root.next = Node(head.next.val)
            has[head.next] = root.next
            head = head.next
            root = root.next
        
        while save:
            if save.random in has:
                save_new.random = has[save.random]
            elif save.random==save1:
                save_new.random = save_new1
            save = save.next
            save_new = save_new.next
        return save_new1


# 146. LRU Cache
# Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
# The functions get and put must each run in O(1) average time complexity.



class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity
        self.list_cache = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        if key in self.list_cache:
            self.list_cache.remove(key)
        self.list_cache.append(key)
        if len(self.list_cache)>self.capacity:
            if self.list_cache[0] in self.cache:
                self.cache.pop(self.list_cache[0])
            self.list_cache = self.list_cache[1:]
        return self.cache[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self.list_cache:
            self.list_cache.remove(key)
        self.list_cache.append(key)
        if len(self.list_cache)>self.capacity:
            if self.list_cache[0] in self.cache:
                self.cache.pop(self.list_cache[0])
            self.list_cache = self.list_cache[1:]
        self.cache[key] = value


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)



# 139. Word Break
# Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
# Note that the same word in the dictionary may be reused multiple times in the segmentation.

class Solution:
    def wordBreak(self, s, wordDict):
        l = len(s)
        n = len(wordDict)
        # if set(s) - set(''.join(wordDict))!=set():
        #     return False
    
        # def dfs(n, s, j, l):
        #     # print('NEW ITERATION ', j, l)
        #     if j==l:
        #         return True
        #     for t in range(n):
        #         if s[j]!=wordDict[t][0]:
        #             continue
        #         k = len(wordDict[t])
        #         if j+k<=l and s[j:j+k]==wordDict[t]:
        #             if dfs(n, s, j+k, l):
        #                 return True
        #     return False

        # return dfs(n,s,0,l)
        flags = [0]*l
        for j in range(l-1, -1, -1):
            for t in range(n):
                if s[j]!=wordDict[t][0]:
                    continue
                k = len(wordDict[t])
                if j+k<=l and s[j:j+k]==wordDict[t]:
                    if (j+k<l and flags[j+k]==1) or j+k>=l:
                        flags[j] = 1
                        break
        # print(flags)
        if flags[0]==1:
            return True
        return False

# 148. Sort List
# Given the head of a linked list, return the list after sorting it in ascending order.

class Solution:
    def sortList(self, head):
        if not head:
            return None

        def split(head):
            mid = head
            fast = head
            prev = None
            while fast and fast.next:
                prev = mid
                mid = mid.next
                fast = fast.next.next
            prev.next = None
            return head, mid

        def mergeSort(arr):
            if arr.next:
                L, R = split(arr)
                # print('SPLIT', L, R)
                L = mergeSort(L)
                R = mergeSort(R)
                # print('MERGE DOESNT WORK HERE ', L, R)
                node = ListNode()
                node_cpy = node

                while L and R:
                    # print(L, R)
                    if L.val<=R.val:
                        # print('OK')
                        node.next = L
                        tmp = L.next
                        L.next = R
                        # R = R.next 
                        L = tmp
                    else:
                        # print('SWITCH')
                        node.next = R
                        tmp = R.next
                        R.next = L
                        # L = L.next
                        R = tmp
                        # print('R rn', R, 'KEY, L too ', L, 'maybe node fine', node)
                    node = node.next
                
                return node_cpy.next
            else:
                return arr

        return mergeSort(head)


# 152. Maximum Product Subarray
# Given an integer array nums, find a subarray that has the largest product, and return the product.
# The test cases are generated so that the answer will fit in a 32-bit integer.

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = max(nums)
        maxs = 1
        mins = 1
        for i in range(len(nums)):
            if nums[i] == 0:
                maxs = 1
                mins = 1
                continue
            maxs, mins = max(maxs * nums[i], mins * nums[i], nums[i]), min(maxs * nums[i], mins * nums[i], nums[i])
            res = max(res, maxs)

        return res

# 189. Rotate Array
# Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        x = k % len(nums)
        # for t in range(1, x+1):
        #     c = nums[-1]
        #     for i in range(len(nums)-1, 0, -1):
        #         nums[i] = nums[i-1]
        #     nums[0] = c
        nums[:x], nums[x:] = nums[-x:], nums[:len(nums)-x]



# 153. Find Minimum in Rotated Sorted Array
# Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:
# [4,5,6,7,0,1,2] if it was rotated 4 times.
# [0,1,2,4,5,6,7] if it was rotated 7 times.
# Given the sorted rotated array nums of unique elements, return the minimum element of this array.

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l = 0 
        r = len(nums)-1
        while l<r-1:
            m = l+(r-l)//2
            # print(l, m, r, nums[l], nums[m], nums[r])
            if nums[m]<=nums[r] and nums[m]<=nums[l] and m-l<=1:
                return nums[m]
            elif nums[l]>nums[m]:
                r = m 
            elif nums[m]>nums[r]:
                l = m
            else: 
                return min(nums[l], nums[r], nums[m])
        return min(nums[l], nums[r])

# 155. Min Stack
# Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
# Implement the MinStack class:
# MinStack() initializes the stack object.
# void push(int val) pushes the element val onto the stack.
# void pop() removes the element on the top of the stack.
# int top() gets the top element of the stack.
# int getMin() retrieves the minimum element in the stack.
# You must implement a solution with O(1) time complexity for each function.

class MinStack:
    def __init__(self):
        self.list = []
        self.minimum = None

    def push(self, val: int) -> None:
        if self.list == [] or val<self.minimum:
            self.minimum = val
        self.list.append(val)

    def pop(self) -> None:
        if self.minimum==self.list[-1]:
            if self.list[:-1]==[]:
                self.minimum = None
            else:
                self.minimum = min(self.list[:-1])
        self.list.pop()

    def top(self) -> int:
        return self.list[-1]

    def getMin(self) -> int:
        return  self.minimum


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()



# 198. House Robber
# You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
# Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

class Solution:
    def rob(self, nums: List[int]) -> int:
        s = 0 
        n = len(nums)
        if n==1:
            return nums[0]
        i = 1
        f = [0] * n
        for i in range(n):
            if i<2:
                f[i] = nums[i]
            elif i==2:
                f[i] = f[0] + nums[i]
            else:
                f[i] = max(f[i-2], f[i-3]) + nums[i]
        return max(f[-1], f[-2])


# 200. Number of Islands
# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
	    
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n, m = len(grid), len(grid[0])
        island = 0
        # for i in range(n):
        #     for j in range(m):
        #         if grid[i][j]=='1':
        #             if s==0:
        #                 s += 1
        #             elif i>0 and j>0 and grid[i-1][j]=='1' and grid[i][j-1]=='1' and grid[i-1][j-1]=='0' and ((j>1 and grid[i][j-2]=='0') or (j<=1)):
        #                 # print('FIRST')
        #                 s -= 1
        #             elif i==1 and j>0 and grid[i-1][j]=='1' and grid[i-1][j-1]=='0':
        #                 # print('SECOND')
        #                 s -= 1
        #             elif j>0 and grid[i][j-1]=='1':
        #                 pass
        #             elif i>0 and grid[i-1][j]=='1':
        #                 pass
        #             else:
        #                 s += 1
        #         # print(i, j, s)
        # return s
        delRow = [1, -1, 0, 0]
        delCol = [0, 0, -1, 1]

        def dfs(grid, vis, n, m, row, col):
            if row < 0 or row >= n or col < 0 or col >= m or vis[row][col] == 1:
                return

            vis[row][col] = 1

            for i in range(4):
                nRow = row + delRow[i]
                nCol = col + delCol[i]

                if 0 <= nRow < n and 0 <= nCol < m and grid[nRow][nCol] == '1' and vis[nRow][nCol] == 0:
                    dfs(grid, vis, n, m, nRow, nCol)
                    
        vis = [[0] * m for _ in range(n)]

        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1' and vis[i][j] == 0:
                    island += 1
                    dfs(grid, vis, n, m, i, j)
        return island


# 199. Binary Tree Right Side View
# Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

class Solution:
    def rightSideView(self, root):
        result = []

        def depth(tree, dpt):
            if not tree:
                return 
            if len(result)==dpt:
                result.append(tree.val)
            depth(tree.right, dpt+1)
            depth(tree.left, dpt+1)
            return 

        depth(root, 0)
        return result

# 215. Kth Largest Element in an Array
# Given an integer array nums and an integer k, return the kth largest element in the array.
# Note that it is the kth largest element in the sorted order, not the kth distinct element.

class Solution:
    def findKthLargest(self, nums, k):
        def mergeSort(arr):
            if len(arr) > 1:
                mid = len(arr)//2
                L = arr[:mid]
                R = arr[mid:]
                mergeSort(L)
                mergeSort(R)
                i = j = k = 0

                while i < len(L) and j < len(R):
                    if L[i] <= R[j]:
                        arr[k] = L[i]
                        i += 1
                    else:
                        arr[k] = R[j]
                        j += 1
                    k += 1

                while i < len(L):
                    arr[k] = L[i]
                    i += 1
                    k += 1

                while j < len(R):
                    arr[k] = R[j]
                    j += 1
                    k += 1
        mergeSort(nums)
        return nums[-k]


# 207. Course Schedule
# There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
# For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
# Return true if you can finish all courses. Otherwise, return false.

class Solution:
    def canFinish(self, numCourses, prerequisites):
        if prerequisites==[]:
            return True
        courses = {}
        for i in prerequisites:
            if i[0]==i[1]:
                return False
            if i[1] not in courses:
                courses[i[1]] = [i[0]]
            else:
                courses[i[1]].append(i[0])
            if i[0] in courses and i[1] in courses[i[0]]:
                return False
                # courses[i[0]].remove(i[1])
                # courses[i[1]].remove(i[0])

        def check_cycle(i, j):
            # print('CYCLE', i, j, visited)
            if i in visited:
                # print('ahh?')
                return False
            visited.append(i)
            for el in j:
                if el in courses:
                    return check_cycle(el, courses[el])
            # return True

        for i, j in courses.items():
            visited = []
            if check_cycle(i, j)==False:
                # print('here')
                return False
        return True


# 230. Kth Smallest Element in a BST
# Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        result = []

        def depth(tree):
            if not tree:
                return 
            depth(tree.left)
            depth(tree.right)
            # print(l, r, tree)
            # if l and r:
            #     r += 1
            # if l==k:
            #     result.append(tree.left.val)
            # if r==k:
            #     result.append(tree.right.val)
            # if r+1==k:
            result.append(tree.val)
        
        depth(root)
        # print(result)
        return sorted(result)[k-1]


# 238. Product of Array Except Self
# Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        p = 1
        zp = 0
        for i in nums:
            if i==0:
                zp += 1
            else:
                p *= i

        for i in range(len(nums)):
            if zp==1 and nums[i]!=0:
                nums[i] = 0
            elif zp>1:
                nums[i] = 0
            elif nums[i]==0:
                nums[i] = p
            else:
                nums[i] = p//nums[i]
        return nums


# 287. Find the Duplicate Number
# Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
# There is only one repeated number in nums, return this repeated number.

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # s = 0
        # k = 0
        # for i in nums:
        #     s += i
        # for t in range(1, len(nums)):
        #     k += t
        # return s-k

        # f = 0
        # i = 0 
        # while True:
        #     if (f==0 and nums[i]==i+1) or (nums[i]==-1):
        #         nums[i] = 0
        #         i = 0 
        #         while nums[i]==0:
        #             i += 1
        #     elif f==0 and nums[i]!=i+1:
        #         nums[i], i = -1, nums[i]-1
        #         f = 1
        #     elif f!=0 and nums[i]==i+1:
        #         return i+1
        #     elif nums[i] == 0:
        #         return i+1
        #     elif nums[i]==i+1 and f==1:
        #         return i+1
        #     else:
        #         nums[i], i = 0, nums[i] - 1
        #     print(nums[i], i)

        # visited = []
        # for i in nums:
        #     if i in visited:
        #         return i
        #     visited.append(i)
        
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i]==nums[i-1]:
                return nums[i]



# 300. Longest Increasing Subsequence
# Given an integer array nums, return the length of the longest strictly increasing subsequence.

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        lis = [1]*n
        for i in range(1, n):
            for j in range(0, i):
                if nums[i] > nums[j] and lis[i] < lis[j] + 1:
                    lis[i] = lis[j]+1
        return max(lis)



	    
