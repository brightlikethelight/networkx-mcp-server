#!/usr/bin/env python3
"""
Test script to demonstrate the efficiency improvement from using deque instead of list.pop(0).
This tests the citation network building performance improvement.
"""

import time
from collections import deque
from typing import List, Tuple

def test_list_pop_performance(n: int) -> float:
    """Test performance using list.pop(0) - the old inefficient way."""
    items = [(f"doi_{i}", 0) for i in range(n)]
    
    start_time = time.time()
    while items:
        current_doi, depth = items.pop(0)
        if depth < 2:
            items.append((f"ref_{current_doi}", depth + 1))
    
    return time.time() - start_time

def test_deque_performance(n: int) -> float:
    """Test performance using deque.popleft() - the new efficient way."""
    items = deque([(f"doi_{i}", 0) for i in range(n)])
    
    start_time = time.time()
    while items:
        current_doi, depth = items.popleft()
        if depth < 2:
            items.append((f"ref_{current_doi}", depth + 1))
    
    return time.time() - start_time

def main():
    """Run performance comparison tests."""
    print("Performance Comparison: list.pop(0) vs deque.popleft()")
    print("=" * 60)
    
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"\nTesting with {size} initial items:")
        
        list_time = test_list_pop_performance(size)
        print(f"  list.pop(0):     {list_time:.4f} seconds")
        
        deque_time = test_deque_performance(size)
        print(f"  deque.popleft(): {deque_time:.4f} seconds")
        
        if deque_time > 0:
            improvement = (list_time - deque_time) / deque_time * 100
            speedup = list_time / deque_time
            print(f"  Improvement:     {improvement:.1f}% faster ({speedup:.1f}x speedup)")
        else:
            print(f"  Improvement:     deque too fast to measure accurately")

if __name__ == "__main__":
    main()
