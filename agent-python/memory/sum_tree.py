"""
SumTree data structure for efficient O(log N) prioritized sampling in Prioritized Experience Replay (PER).
"""

from typing import Tuple
import numpy as np


class SumTree:
    """Binary tree where each parent node contains the sum of its children."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree array has size 2 * capacity - 1
        # Indices 0 .. capacity-2 are parent nodes
        # Indices capacity-1 .. 2*capacity-2 are leaf nodes storing priorities
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float):
        """Adds a priority for a new transition at data_pointer."""
        tree_index = self.data_pointer + self.capacity - 1
        self.update(tree_index, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index: int, priority: float):
        """Updates priority at tree_index and propagates change up the tree."""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate changes up to root
        parent = (tree_index - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """
        Traverses tree from root to leaf to locate the segment containing 'value'.
        Returns (tree_index, priority, data_index).
        """
        parent = 0
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1

            # Reached a leaf node
            if left_child >= len(self.tree):
                tree_index = parent
                break

            if value <= self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child

        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], data_index

    @property
    def total_priority(self) -> float:
        """Returns the sum of all priorities (root value)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns maximum priority stored among all active leaves."""
        leaves = self.tree[self.capacity - 1: self.capacity - 1 + self.size]
        if len(leaves) == 0 or np.all(leaves == 0):
            return 1.0
        return float(np.max(leaves))

