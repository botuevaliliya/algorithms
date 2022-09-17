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

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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

