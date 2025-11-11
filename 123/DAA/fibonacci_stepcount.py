# Program: Fibonacci Series (Non-Recursive and Recursive) with Step Count

# Global step counters
step_iter = 0
step_recur = 0

# ---------------- Non-Recursive Fibonacci ----------------
def fibonacci_iterative(n):
    global step_iter
    a, b = 0, 1
    step_iter += 1  # Initialization step

    for i in range(2, n + 1):
        step_iter += 1  # For loop iteration count
        a, b = b, a + b
        step_iter += 1  # Assignment step
    return b if n > 0 else a


# ---------------- Recursive Fibonacci ----------------
def fibonacci_recursive(n):
    global step_recur
    step_recur += 1  # Function call count
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


# ---------------- Main Program ----------------
N = 10  # You can change this value
print(f"Given N = {N}")

# Non-recursive call
fib_iter = fibonacci_iterative(N)
print(f"Fibonacci (Non-Recursive): {fib_iter}")
print(f"Step Count (Non-Recursive): {step_iter}")

# Recursive call
fib_recur = fibonacci_recursive(N)
print(f"Fibonacci (Recursive): {fib_recur}")
print(f"Step Count (Recursive): {step_recur}")
