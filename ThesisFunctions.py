# Required libraries for mathematical operations, data manipulation, and plotting.
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy.plotting as spp
from sympy import symbols, diff, Poly, solve
from mpl_toolkits.mplot3d import Axes3D

def model(t, x0, alpha):
    """
    Exponential growth model.
    Computes x0 * exp(alpha * t).

    Parameters:
    - t (array-like): Time variable(s) at which to evaluate the model.
    - x0 (float): Initial value at time t=0.
    - alpha (float): Growth rate parameter.

    Returns:
    - array: Model predictions at each time point in t.
    """
    exp_input = float(alpha) * np.array(t, dtype=float)
    return x0 * np.exp(exp_input)

def sse(params, data):
    """
    Sum of squared errors (SSE) between model predictions and data.

    Parameters:
    - params (tuple): Model parameters (x0, alpha).
    - data (pd.DataFrame): Observed data with columns 'Time' and 'Data'.

    Returns:
    - float: Calculated SSE.
    """
    x0, alpha = params
    predictions = model(data['Time'], x0, alpha)
    return np.sum((data['Data'] - predictions) ** 2)



def data_gen(num_data_points=4, noise_level=0.1, alpha=2, x0=1):
    """
    Generates synthetic data based on the exponential growth model with added noise,
    where the noise is a certain percentage of the data value, distributed normally.

    Parameters:
    - num_data_points (int): Number of data points to generate.
    - noise_level (float/int): Proportional standard deviation of the Gaussian noise as a percentage of each data point value.
    - alpha (float/int): True growth rate parameter used for data generation.
    - x0 (float/int): True initial value used for data generation.

    Returns:
    - pd.DataFrame: Generated data containing 'Time' and 'Data' columns.
    """
    t = np.arange(num_data_points)
    x = x0 * np.exp(alpha * t)
    # Generate noise as a percentage of x, ensuring it does not exceed x itself
    
    noise = noise_level * x *  np.random.normal(0, 1, size=t.shape) * 0.1

    # Ensuring noise does not exceed the value of x
    #noise = np.clip(noise, -x/4, x/4)
    x_noisy = x + noise    

    # Convert noisy data to a rational number with a maximum limit
    x_noisy_rational = np.array([min(sp.Rational(int(xn * 100), 100), 10000) for xn in x_noisy])
    data = pd.DataFrame({'Time': t, 'Data': x_noisy_rational})
    return data



# def data_gen(num_data_points=4, noise_level=0.1, alpha=2, x0=1):
#     """
#     Generates synthetic data based on the exponential growth model with added noise.

#     Parameters:
#     - num_data_points (int): Number of data points to generate.
#     - noise_level (float/int): Standard deviation of the Gaussian noise.
#     - alpha (float/int): True growth rate parameter used for data generation.
#     - x0 (float/int): True initial value used for data generation.

#     Returns:
#     - pd.DataFrame: Generated data containing 'Time' and 'Data' columns.
#     """
#     t = np.arange(num_data_points)
#     x = x0 * np.exp(alpha * t)
#     noise = np.random.normal(0, noise_level, size=t.shape)
#     x_noisy = x + noise

#     x_noisy_rational = np.array([min(sp.Rational(int(xn * 100), 100), 10000) for xn in x_noisy])
#     data = pd.DataFrame({'Time': t, 'Data': x_noisy_rational})
#     #print(data)
#     return data

def run_optimization(num_runs, initial_guess, data, method='L-BFGS-B', bounds=None):
    """
    Performs multiple runs of numerical optimization to estimate model parameters.

    Parameters:
    - num_runs (int): Number of optimization attempts.
    - initial_guess (tuple): Initial guess for the model parameters (x0, alpha).
    - data (pd.DataFrame): Observed data with columns 'Time' and 'Data'.
    - method (str): Optimization method.
    - bounds (list of tuples): Bounds on the parameters.

    Returns:
    - np.array: Array of estimated parameters from each successful optimization run.
    """
    final_params = []
    for _ in range(num_runs):
        result = minimize(sse, initial_guess, args=(data), method=method, bounds=bounds)
        if result.success:
            final_params.append(result.x)
        else:
            final_params.append(None)
            print(f"Optimization failed: {result.message} for (x0, alpha) = {initial_guess}")
    return np.array([param for param in final_params if param is not None])

def error_matrix(alpha_range, x0_range, data, num_runs=1):
    """
    Constructs an error matrix for different combinations of alpha and x0.

    Parameters:
    - alpha_range (list): List of alpha values to test.
    - x0_range (list): List of x0 values to test.
    - data (pd.DataFrame): Observed data with columns 'Time' and 'Data'.
    - num_runs (int): Number of optimization runs for each parameter set.

    Returns:
    - np.ndarray: Matrix of SSE values for each (alpha, x0) pair.
    """
    error_matrix = np.zeros((len(alpha_range), len(x0_range)))
    for i, alpha in enumerate(alpha_range):
        for j, x0 in enumerate(x0_range):
            initial_guess = (x0, alpha)
            final_param = run_optimization(num_runs, initial_guess, data)
            error_matrix[i, j] = sse(final_param[0], data) if len(final_param) > 0 else np.inf
    return error_matrix


def evaluate_hessian_at_extremas(params, x_i, t_i, epsilon = 0.001):
    """
    Evaluates the Hessian matrix at the estimated extremas to determine if they represent maxima.

    Parameters:
    - params (list of tuples): Estimated parameters (x0, alpha).
    - x_i (list): Observed data values.
    - t_i (list): Corresponding time values for the observed data.

    Returns:
    - list of tuples: Each tuple contains the parameter set and a boolean indicating if it is a maximum.
    """
    x0, b = sp.symbols('x0 b')
    SSE_poly = sum([(x - x0 * b**t)**2 for x, t in zip(x_i, t_i)])
    partial_x0x0 = sp.diff(SSE_poly, x0, x0)
    partial_x0b = sp.diff(SSE_poly, x0, b)
    partial_bb = sp.diff(SSE_poly, b, b)
    hessian_general = sp.Matrix([[partial_x0x0, partial_x0b], [partial_x0b, partial_bb]])
    maxima_results = []
    for param in params:
        #if param[0] < epsilon:
        #   maxima_results.append((param, True))
        #else:
        hessian_at_point = hessian_general.subs({x0: param[0], b: np.exp(param[1])})
        #print(hessian_at_point)
        det = hessian_at_point.det()
        trace = hessian_at_point.trace()
        if det > 0 and trace > 0:
            maxima_results.append((param, True))
        else:
            print("hessian is zero")
            B = groeb(t_i, x_i)
            params_ = find_roots_alternative(B[1])
            print("rec")
            evaluate_hessian_at_extremas(params, x_i, t_i)
    return maxima_results

def find_roots_alternative(poly):
    b = sp.symbols('b')
    
    # Convert sympy polynomial to a function for Newton's method
    poly_func = sp.lambdify(b, poly, 'numpy')
    poly_derivative = sp.diff(poly, b)
    poly_derivative_func = sp.lambdify(b, poly_derivative, 'numpy')
    
    # Call sturm algo
    num_positive_roots = count_positive_roots(poly)

    # Call Newton Method
    positive_roots = find_roots_newton(poly_func, poly_derivative_func, num_positive_roots)

    return positive_roots


# Newton's method to find roots
def find_roots_newton(func, func_prime, num_roots, initial_guess_range=(0, 10), max_attempts_per_root=100):
    roots_found = []
    for _ in range(num_roots):
        root_found = False
        attempts = 0
        while not root_found and attempts < max_attempts_per_root:
            initial_guess = np.random.uniform(*initial_guess_range)
            try:
                root = newton(func, initial_guess, fprime=func_prime, maxiter=50)
                if root>0 and all(abs(root - found_root) > 1e-5 for found_root in roots_found):
                    roots_found.append(root)
                    root_found = True
            except RuntimeError:
                pass  # Handle case where Newton's method fails
            attempts += 1
        if not root_found:
            raise ValueError("Failed to find all positive roots after specified attempts.")
    
    
    #Making sure the roots are positive MISLEADING?
    
    
    ret = [root for root in roots_found if root>0]
    return ret


def find_x0_alpha_pairs(G, b_arr):
    """
    Finds pairs of (x0, alpha) that satisfy the system of equations G for given values of b.

    Parameters:
    - G (list of sympy expressions): System of equations to solve.
    - b_arr (list of floats): Values of b to plug into the system.

    Returns:
    - list of tuples: Each tuple represents a pair of (x0, alpha) that solves the system for a given b.
    """
    x0, b = sp.symbols('x0 b')
    x0_arr = []
    alpha_arr = []
    for b_i in b_arr:
        x0_i = sp.solve(G[0].subs(b, b_i), x0)[0]
        x0_arr.append(x0_i)
        alpha_arr.append(sp.log(b_i))
    return [(float(x0), float(alpha)) for x0, alpha in zip(x0_arr, alpha_arr)]

def roots_symbolic(poly):
    """
    Finds positive, real roots of a symbolic polynomial.

    Parameters:
    - poly (sympy expression): Polynomial for which to find roots.

    Returns:
    - list of floats: Positive, real roots of the polynomial.
    """
    x = symbols('b')
    roots = solve(poly, x)
    return [root.evalf() for root in roots if root.is_real and root > 0]


def sturm_sequence(p):
    """
    Generates the Sturm sequence for a given polynomial.

    Parameters:
    - p (sympy.Poly): Polynomial for which to generate the Sturm sequence.

    Returns:
    - list: List of polynomials representing the Sturm sequence.
    """
    x0, b = sp.symbols('x0 b')
    p0 = p
    p1 = diff(p, b)
    sturm_seq = [p0, p1]
    # Generate the rest of the Sturm sequence using sympy's polynomial division
    while not sturm_seq[-1].is_zero:
        p_i, remainder = sturm_seq[-2].as_expr(), sturm_seq[-1].as_expr()
        div_result = Poly(p_i, b).div(Poly(remainder, b))
        sturm_seq.append(-div_result[1])
    # Remove the last polynomial if it's zero.
    if sturm_seq[-1].is_zero:
        sturm_seq.pop()
    return [p.as_expr() for p in sturm_seq]

def count_sign_changes(sequence, value):
    """
    Counts the number of sign changes in a Sturm sequence at a given value.
    Handles NaN and complex numbers by ignoring them or treating them according to research needs.

    Parameters:
    - sequence (list): Sturm sequence as a list of sympy expressions.
    - value (float): Value at which to evaluate the sign changes.

    Returns:
    - int: Number of sign changes in the sequence at the given value.
    """
    x0, b = sp.symbols('x0 b')
    signs = [p.subs(b, value) for p in sequence]
    sign_changes = 0
    previous_sign = None
    for sign in signs:
        current_sign = sign > 0
        if previous_sign is not None and current_sign != previous_sign:
            sign_changes += 1
        previous_sign = current_sign
    return sign_changes


def count_positive_roots(p):
    """
    Uses the Sturm sequence method to count the number of positive roots of a polynomial.

    Parameters:
    - p (sympy.Poly): Polynomial for which to count the positive roots.

    Returns:
    - int: Number of positive roots of the polynomial.
    """
    x0, b = sp.symbols('x0 b')
    sturm_seq = sturm_sequence(Poly(p, b))
    # Count sign changes at positive infinity and zero.
    sign_changes_at_inf = count_sign_changes(sturm_seq, 1e10)  # Simulate positive infinity.
    sign_changes_at_zero = count_sign_changes(sturm_seq, 0) #INCLUDING 0
    return sign_changes_at_zero - sign_changes_at_inf 

def groeb(x_i, t_i):
    """
    Calculates the Groebner basis for the system of equations derived from the SSE's partial derivatives.

    Parameters:
    - x_i (numpy.ndarray): Observed data values.
    - t_i (numpy.ndarray): Corresponding time points.

    Returns:
    - list: Groebner basis of the system, facilitating the solution for model parameters x0 and b.
    
    The Groebner basis is computed for the system formed by the partial derivatives of the SSE,
    with respect to the initial value x0 and growth factor b, of an exponential growth model.
    """
    x0, b = sp.symbols('x0 b')
    SSE_poly = sum([(x - x0 * b**t)**2 for x, t in zip(x_i, t_i)])
    partial_x0 = sp.diff(SSE_poly, x0)
    partial_b = sp.diff(SSE_poly, b)
    B = sp.groebner([partial_x0, partial_b], x0, b, order='lex')
    
    return B
