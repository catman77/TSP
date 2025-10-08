#!/usr/bin/env python3
"""
COMPREHENSIVE TESTING: categorical_tsp_v3.py on 176 graphs
===========================================================

Testing on the same graphs used in test_extended_categorical.py:
- Random graphs (90 tests)
- Chaotic graphs (50 tests)
- Pathological graphs (36 tests)

Total: 176 tests

Goal: Verify that categorical algorithm v3 works correctly
and finds optimal solution on all graph types.
"""

import sys
import time
import random
from typing import List, Tuple, Dict
from itertools import permutations

sys.path.append('/home/catman/Yandex.Disk/cuckoo/z/reals/libs/Experiments/Info/article/exp_tsp/paper_latex')

from categorical_tsp_v3 import solve_tsp_categorical


# ============================================================================
# Graph generators (from test_extended_categorical.py)
# ============================================================================

def generate_random_graph(n: int, seed: int, min_weight: int = 1, max_weight: int = 20) -> List[List[int]]:
    """RANDOM GRAPH (as in Phase 2)"""
    random.seed(seed)
    return [[0 if i == j else random.randint(min_weight, max_weight) for j in range(n)] for i in range(n)]


def generate_chaotic_graph(n: int, seed: int, chaos_type: str = 'extreme_variance') -> List[List[int]]:
    """CHAOTIC GRAPH with pathological properties"""
    random.seed(seed)
    W = [[0] * n for _ in range(n)]
    
    if chaos_type == 'extreme_variance':
        for i in range(n):
            for j in range(i + 1, n):
                w = random.choice([1, 2, 3] + [random.randint(100, 1000)])
                W[i][j] = W[j][i] = w
    
    elif chaos_type == 'prime_weights':
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for i in range(n):
            for j in range(i + 1, n):
                w = random.choice(primes)
                W[i][j] = W[j][i] = w
    
    elif chaos_type == 'exponential':
        for i in range(n):
            for j in range(i + 1, n):
                k = i * n + j
                w = 2 ** (k % 10)
                W[i][j] = W[j][i] = w
    
    elif chaos_type == 'fibonacci':
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        for i in range(n):
            for j in range(i + 1, n):
                k = (i * n + j) % len(fib)
                W[i][j] = W[j][i] = fib[k]
    
    elif chaos_type == 'adversarial':
        for i in range(n):
            for j in range(i + 1, n):
                if i == 0:
                    W[i][j] = W[j][i] = 1
                elif j == n - 1:
                    W[i][j] = W[j][i] = 10
                else:
                    W[i][j] = W[j][i] = 100
    
    return W


def generate_pathological_graph(n: int, pattern: str = 'metric_trap') -> List[List[int]]:
    """PATHOLOGICAL GRAPH"""
    W = [[0] * n for _ in range(n)]
    
    if pattern == 'metric_trap':
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) == 1:
                    w = 1
                elif abs(i - j) == n - 1:
                    w = 50
                else:
                    w = 5
                W[i][j] = W[j][i] = w
    
    elif pattern == 'cyclic_anomaly':
        for i in range(n):
            for j in range(i + 1, n):
                if (i + j) % 2 == 0:
                    w = abs(i - j) + 1
                else:
                    w = abs(i - j) + 10
                W[i][j] = W[j][i] = w
    
    elif pattern == 'bridge':
        half = n // 2
        for i in range(n):
            for j in range(i + 1, n):
                if i < half and j < half:
                    w = 2
                elif i >= half and j >= half:
                    w = 2
                elif abs(i - half) <= 1 and abs(j - half) <= 1:
                    w = 5
                else:
                    w = 50
                W[i][j] = W[j][i] = w
    
    return W


# ============================================================================
# Helper functions
# ============================================================================

def matrix_to_graph(W: List[List[int]]) -> Dict[int, Dict[int, float]]:
    """Convert weight matrix to graph format for categorical_tsp_v3"""
    n = len(W)
    graph = {}
    for i in range(n):
        graph[i] = {}
        for j in range(n):
            if i != j and W[i][j] > 0:
                graph[i][j] = float(W[i][j])
    return graph


def brute_force_tsp(W: List[List[int]]) -> Tuple[List[int], float]:
    """Find optimum by enumeration (for small n)"""
    n = len(W)
    vertices = list(range(1, n))  # Fix start = 0
    
    best_cost = float('inf')
    best_path = None
    
    for perm in permutations(vertices):
        path = [0] + list(perm) + [0]
        cost = sum(W[path[i]][path[i+1]] for i in range(len(path) - 1))
        if cost < best_cost:
            best_cost = cost
            best_path = path
    
    return best_path, best_cost


# ============================================================================
# Testing
# ============================================================================

def test_on_graph(W: List[List[int]], description: str, test_id: int, verbose: bool = True) -> Dict:
    """Test on a single graph"""
    n = len(W)
    
    # Find optimum by brute force
    opt_path, opt_cost = brute_force_tsp(W)
    
    # Test categorical_tsp_v3
    graph = matrix_to_graph(W)
    
    try:
        start_time = time.time()
        cat_path, cat_cost = solve_tsp_categorical(graph, start=0)
        elapsed = time.time() - start_time
        
        optimal = abs(cat_cost - opt_cost) < 0.01
        
        if verbose:
            status = "✅" if optimal else "❌"
            print(f"\nTest #{test_id}: {description}")
            print(f"  n={n}, optimum={opt_cost:.0f}, categorical={cat_cost:.0f} {status}")
            if not optimal:
                print(f"  ⚠️  Deviation: {cat_cost - opt_cost:.2f} ({100*(cat_cost/opt_cost - 1):.1f}%)")
            print(f"  Time: {elapsed:.4f}s")
        
        return {
            'test_id': test_id,
            'description': description,
            'n': n,
            'optimal_cost': opt_cost,
            'categorical_cost': cat_cost,
            'is_optimal': optimal,
            'time': elapsed
        }
    
    except Exception as e:
        print(f"\n❌ Test #{test_id} FAILED: {description}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'test_id': test_id,
            'description': description,
            'n': n,
            'optimal_cost': opt_cost,
            'categorical_cost': float('inf'),
            'is_optimal': False,
            'time': 0,
            'error': str(e)
        }


def run_all_tests():
    """Run all 176+ tests"""
    print("="*100)
    print(" " * 25 + "COMPREHENSIVE TESTING categorical_tsp_v3")
    print("="*100)
    print("\nGoal: Verify correctness on 176+ graphs of different types")
    print("Types: Random (9×10=90), Chaotic (5×10=50), Pathological (3×12=36)")
    print()
    
    results = []
    test_id = 1
    
    # ========================================================================
    # BLOCK 1: RANDOM GRAPHS (9 configurations × 10 repetitions = 90 tests)
    # ========================================================================
    
    print("\n" + "="*100)
    print("BLOCK 1: RANDOM GRAPHS (90 tests)")
    print("="*100)
    print("Configurations: n∈{6,7,8} × base_seeds∈{1,2,3}")
    print("Repetitions: each configuration × 10 different seeds")
    
    test_count = 0
    for n in [6, 7, 8]:
        for base_seed in [1, 2, 3]:
            for rep in range(10):
                seed = base_seed * 1000 + rep
                W = generate_random_graph(n, seed, min_weight=1, max_weight=20)
                desc = f"Random n={n}, base={base_seed}, rep={rep}"
                
                result = test_on_graph(W, desc, test_id, verbose=(rep == 0))
                results.append(result)
                test_id += 1
                test_count += 1
    
    random_results = results[-test_count:]
    accuracy = sum(1 for r in random_results if r['is_optimal']) / len(random_results) * 100
    avg_time = sum(r['time'] for r in random_results) / len(random_results)
    
    print(f"\n📊 STATISTICS (Random graphs):")
    print(f"   Accuracy: {sum(1 for r in random_results if r['is_optimal'])}/{len(random_results)} ({accuracy:.1f}%)")
    print(f"   Average time: {avg_time:.4f}s")
    
    # ========================================================================
    # BLOCK 2: CHAOTIC GRAPHS (5 types × 10 repetitions = 50 tests)
    # ========================================================================
    
    print("\n" + "="*100)
    print("BLOCK 2: CHAOTIC GRAPHS (50 tests)")
    print("="*100)
    print("Types: extreme_variance, prime_weights, exponential, fibonacci, adversarial")
    print("Each type × 10 different seeds")
    
    chaos_types = [
        ('extreme_variance', "Extreme variance"),
        ('prime_weights', "Prime numbers"),
        ('exponential', "Exponential growth"),
        ('fibonacci', "Fibonacci"),
        ('adversarial', "Anti-greedy")
    ]
    
    test_count = 0
    for chaos_type, type_name in chaos_types:
        for rep in range(10):
            n = 6  # Fix n=6 for stability
            seed = rep + 1
            W = generate_chaotic_graph(n, seed, chaos_type)
            desc = f"Chaotic: {type_name}, rep={rep}"
            
            result = test_on_graph(W, desc, test_id, verbose=(rep == 0))
            results.append(result)
            test_id += 1
            test_count += 1
    
    chaotic_results = results[-test_count:]
    accuracy = sum(1 for r in chaotic_results if r['is_optimal']) / len(chaotic_results) * 100
    avg_time = sum(r['time'] for r in chaotic_results) / len(chaotic_results)
    
    print(f"\n📊 STATISTICS (Chaotic graphs):")
    print(f"   Accuracy: {sum(1 for r in chaotic_results if r['is_optimal'])}/{len(chaotic_results)} ({accuracy:.1f}%)")
    print(f"   Average time: {avg_time:.4f}s")
    
    # ========================================================================
    # BLOCK 3: PATHOLOGICAL GRAPHS (3 patterns × 12 sizes = 36 tests)
    # ========================================================================
    
    print("\n" + "="*100)
    print("BLOCK 3: PATHOLOGICAL GRAPHS (36 tests)")
    print("="*100)
    print("Patterns: metric_trap, cyclic_anomaly, bridge")
    print("Sizes: n∈{4,5,6,7} × each pattern")
    
    patterns = [
        ('metric_trap', "Metric trap"),
        ('cyclic_anomaly', "Cyclic anomaly"),
        ('bridge', "Bridge between clusters")
    ]
    
    test_count = 0
    for pattern, pattern_name in patterns:
        for n in [4, 5, 6, 7]:
            for rep in range(3):  # 3 repetitions for each n
                W = generate_pathological_graph(n, pattern)
                # Slightly vary weights for diversity
                if rep > 0:
                    for i in range(n):
                        for j in range(i+1, n):
                            W[i][j] = W[j][i] = W[i][j] + rep
                
                desc = f"Pathological: {pattern_name}, n={n}, var={rep}"
                
                result = test_on_graph(W, desc, test_id, verbose=(rep == 0 and n == 4))
                results.append(result)
                test_id += 1
                test_count += 1
    
    pathological_results = results[-test_count:]
    accuracy = sum(1 for r in pathological_results if r['is_optimal']) / len(pathological_results) * 100
    avg_time = sum(r['time'] for r in pathological_results) / len(pathological_results)
    
    print(f"\n📊 STATISTICS (Pathological graphs):")
    print(f"   Accuracy: {sum(1 for r in pathological_results if r['is_optimal'])}/{len(pathological_results)} ({accuracy:.1f}%)")
    print(f"   Average time: {avg_time:.4f}s")
    
    # ========================================================================
    # FINAL STATISTICS
    # ========================================================================
    
    print("\n" + "="*100)
    print(" " * 35 + "FINAL STATISTICS")
    print("="*100)
    
    total_tests = len(results)
    total_optimal = sum(1 for r in results if r['is_optimal'])
    total_accuracy = total_optimal / total_tests * 100
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / total_tests
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   Total tests: {total_tests}")
    print(f"   Optimal solutions: {total_optimal}/{total_tests} ({total_accuracy:.2f}%)")
    print(f"   Average time per test: {avg_time:.4f}s")
    print(f"   Total time: {total_time:.2f}s")
    
    print(f"\n📊 BY BLOCKS:")
    print(f"   Random graphs:      {sum(1 for r in random_results if r['is_optimal'])}/{len(random_results)} "
          f"({sum(1 for r in random_results if r['is_optimal'])/len(random_results)*100:.1f}%)")
    print(f"   Chaotic graphs:     {sum(1 for r in chaotic_results if r['is_optimal'])}/{len(chaotic_results)} "
          f"({sum(1 for r in chaotic_results if r['is_optimal'])/len(chaotic_results)*100:.1f}%)")
    print(f"   Pathological graphs: {sum(1 for r in pathological_results if r['is_optimal'])}/{len(pathological_results)} "
          f"({sum(1 for r in pathological_results if r['is_optimal'])/len(pathological_results)*100:.1f}%)")
    
    # Check failures
    failures = [r for r in results if not r['is_optimal']]
    if failures:
        print(f"\n⚠️  FAILED TESTS ({len(failures)}):")
        for r in failures[:10]:  # Show first 10
            print(f"   #{r['test_id']}: {r['description']}")
            print(f"      Optimum={r['optimal_cost']:.0f}, Got={r['categorical_cost']:.0f}, "
                  f"Diff={r['categorical_cost'] - r['optimal_cost']:.2f}")
    
    print("\n" + "="*100)
    if total_accuracy == 100.0:
        print(" " * 35 + "🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f" " * 30 + f"⚠️  Accuracy: {total_accuracy:.2f}% ({total_tests - total_optimal} errors)")
    print("="*100)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
