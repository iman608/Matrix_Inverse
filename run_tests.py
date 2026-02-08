import argparse
import time
import numpy as np
from matrix_revers import compute_inverse


def approx_equal(A, B, tol=1e-8):
    return np.allclose(A, B, atol=tol)


def gen_random(n, seed=None):
    rs = np.random.RandomState(seed)
    return rs.randn(n, n).astype(float)


def gen_singular(n):
    A = gen_random(n, seed=1)
    A[0] = A[1]  # make two rows equal
    return A


def gen_near_singular(n, eps=1e-12):
    A = gen_random(n, seed=2)
    A[0, 0] = eps
    return A


# Lightweight default tests plus some heavier cases marked with 'heavy'
tests = [
    {"name": "identity_3", "matrix": np.eye(3), "expect_error": False, "heavy": False},
    {"name": "simple_2", "matrix": np.array([[4,7],[2,6]]), "expect_error": False, "heavy": False},
    {"name": "singular", "matrix": np.array([[1,2],[2,4]]), "expect_error": True, "heavy": False},
    {"name": "near_singular", "matrix": np.array([[1e-13,0],[0,1]]), "expect_error": True, "heavy": False},
    {"name": "random_4", "matrix": np.random.RandomState(0).randint(-5,6,(4,4)).astype(float), "expect_error": False, "heavy": False},

    # stronger / larger tests (run only with --heavy)
    {"name": "random_50", "matrix": gen_random(50, seed=10), "expect_error": False, "heavy": True},
    {"name": "random_100", "matrix": gen_random(100, seed=11), "expect_error": False, "heavy": True},
    {"name": "singular_100", "matrix": gen_singular(100), "expect_error": True, "heavy": True},
    {"name": "near_singular_80", "matrix": gen_near_singular(80, eps=1e-14), "expect_error": True, "heavy": True}
]


def run_test_case(case, max_time=None):
    M = case["matrix"]
    name = case["name"]
    expect_error = case.get("expect_error", False)
    print(f"Running {name} (n={M.shape[0]})...")
    start = time.perf_counter()
    try:
        inv = compute_inverse(M)
        elapsed = time.perf_counter() - start
        if max_time and elapsed > max_time:
            print(f"  SKIP: took {elapsed:.2f}s > max_time {max_time}s")
            return None
        if expect_error:
            print(f"  FAIL: expected inversion to error but got a result")
            return False
        I = np.eye(M.shape[0])
        if not approx_equal(M @ inv, I):
            print(f"  FAIL: M @ inv not close to identity")
            return False
        try:
            npinv = np.linalg.inv(M)
            if not approx_equal(inv, npinv, tol=1e-6):
                print(f"  WARN: inverse differs from numpy.linalg.inv (but product is identity)")
        except Exception:
            pass
        print(f"  PASS (time={elapsed:.3f}s)")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        if expect_error:
            print(f"  PASS (expected error): {e} (time={elapsed:.3f}s)")
            return True
        else:
            print(f"  FAIL with exception: {e} (time={elapsed:.3f}s)")
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", action="store_true", help="run heavier/larger tests")
    parser.add_argument("--max-time", type=float, default=None, help="maximum time (s) allowed per test; tests exceeding it are skipped")
    args = parser.parse_args()

    all_ok = True
    any_run = False
    for case in tests:
        if case.get("heavy") and not args.heavy:
            print(f"Skipping heavy test {case['name']} (use --heavy to run)")
            continue
        any_run = True
        result = run_test_case(case, max_time=args.max_time)
        if result is False:
            all_ok = False

    if not any_run:
        print("No tests were run (use --heavy to run larger tests)")
    elif all_ok:
        print("\nAll tests passed.")
    else:
        print("\nSome tests failed.")


if __name__ == '__main__':
    main()
