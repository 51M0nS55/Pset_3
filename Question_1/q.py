import numpy as np
from scipy.integrate import nquad

# ------------------------------
# 🟢 Helper Functions
# ------------------------------

def is_positive_definite(A):
    """Check if matrix A is positive definite."""
    return np.all(np.linalg.eigvals(A) > 0)

def gaussian_integrand(*v, A, w):
    """Computes the exponent of the Gaussian function."""
    v = np.array(v)
    return np.exp(-0.5 * np.dot(v, np.dot(A, v)) + np.dot(w, v))

def compute_integral(A, w):
    """Computes the N-dimensional integral numerically using quadrature."""
    N = len(w)  # Dimension of the integral
    bounds = [(-np.inf, np.inf)] * N  # Integrate over entire space

    # Use lambda to ensure function signature matches expected input
    integral, error = nquad(lambda *args: gaussian_integrand(*args, A=A, w=w), bounds)
    
    return integral

def closed_form_solution(A, w):
    """Computes the closed-form solution of the Gaussian integral."""
    A_inv = np.linalg.inv(A)
    det_A_inv = np.linalg.det(A_inv)
    exponent = 0.5 * np.dot(w.T, np.dot(A_inv, w))
    return np.sqrt((2 * np.pi) ** len(A) * det_A_inv) * np.exp(exponent)

def compute_moments(A, w):
    """Compute first and second moments using A^-1."""
    A_inv = np.linalg.inv(A)
    first_moment = np.dot(A_inv, w)
    second_moment = A_inv + np.outer(first_moment, first_moment)
    return first_moment, second_moment



if __name__ == "__main__":
    # Given Matrices
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
    A_prime = np.array([[4, 2, 1], [2, 1, 3], [1, 3, 6]])  # Another matrix for comparison
    w = np.array([1, 2, 3])

    # ✅ Check Positive Definiteness
    print(f"Is A positive definite? {is_positive_definite(A)}")
    print(f"Is A' positive definite? {is_positive_definite(A_prime)}")

    # ✅ Compute Integrals
    integral_A = compute_integral(A, w)
    closed_A = closed_form_solution(A, w)

    print("\n🔹 Numerical Integral for A:", integral_A)
    print("🔹 Closed-form Solution for A:", closed_A)
    
    integral_A_prime = compute_integral(A_prime, w)
    closed_A_prime = closed_form_solution(A_prime, w)

    print("\n🔹 Numerical Integral for A':", integral_A_prime)
    print("🔹 Closed-form Solution for A':", closed_A_prime)

    # ✅ Compute Moments
    first_moment, second_moment = compute_moments(A, w)
    
    print("\n🟢 First Moments:", first_moment)
    print("🟢 Second Moments Matrix:\n", second_moment)

    # ✅ Extracting Moments
    v1, v2, v3 = first_moment
    v1v2, v2v3, v1v3 = second_moment[0, 1], second_moment[1, 2], second_moment[0, 2]

    print(f"\n🔹 <v1> = {v1}, <v2> = {v2}, <v3> = {v3}")
    print(f"🔹 <v1 v2> = {v1v2}, <v2 v3> = {v2v3}, <v1 v3> = {v1v3}")

    # ✅ Higher order moments
    v1_squared_v2 = v1 ** 2 * v2
    v2_squared_v3_squared = v2 ** 2 * v3 ** 2

    print(f"\n🔹 <v1^2 v2> = {v1_squared_v2}")
    print(f"🔹 <v2^2 v3^2> = {v2_squared_v3_squared}")