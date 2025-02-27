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
        hessian_at_point = hessian_general.subs({x0: param[0], b: np.exp(param[1])})
        #print(hessian_at_point)
        det = hessian_at_point.det()
        trace = hessian_at_point.trace()

        if det > 0 and trace > 0:
            maxima_results.append((param, True))
        else:
            maxima_results.append((param, False))
    #s = sum(1 for _, is_true in maxima_results if is_true)
    #if s == 0:
        #for param in params:
            #print("No minimas found, checking Gradient")
            #print(sp.diff(SSE_poly, x0).subs({x0: param[0], b: np.exp(param[1])}))
            #print(sp.diff(SSE_poly, b).subs({x0: param[0], b: np.exp(param[1])}))
            #print(maxima_results)
    return maxima_results

def find_roots_alternative(poly):
    """
    An alternative method to identify the roots of a given polynomial by first applying Sturm's theorem
    to count the number of positive roots and then employing Newton's method to accurately find these roots.
    This function demonstrates an innovative approach to root finding, especially when dealing with polynomials
    where traditional methods might struggle.

    Parameters:
    - poly (sympy expression): The polynomial for which positive roots are to be found.

    Returns:
    - list: A list of positive roots found using Newton's method, guided by the initial count obtained
      through Sturm's theorem.
    """
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

from scipy.optimize import newton


# Newton's method to find roots
def find_roots_newton(func, func_prime, num_roots, initial_guess_range=(-10, 10), max_attempts_per_root=100):
    """
    Utilizes Newton's method to find positive roots of a given function.

    This function attempts to find a specified number of positive roots for a given function
    using Newton's method. It iterates through random initial guesses within a specified range
    and applies Newton's method to find roots. If a positive root is found that is distinct 
    (within a threshold) from previously found roots, it is added to the list of roots.

    Parameters:
    - func (callable): The function for which roots are being found.
    - func_prime (callable): The derivative of the function.
    - num_roots (int): The number of positive roots to find.
    - initial_guess_range (tuple of float): The range (min, max) for the initial guess for Newton's method.
    - max_attempts_per_root (int): Maximum number of attempts to find each root.

    Returns:
    - list of float: A list of the positive roots found.

    Raises:
    - ValueError: If it fails to find the specified number of positive roots after the given number of attempts.
    """
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
            return []
    
    ret = [root for root in roots_found if root>0]
    return ret

def grid_search_around_extrema(extrema, data, grid_size=12, step_size=0.01):
    """
    Performs a grid search around an extrema to find a local minimum.

    Parameters:
    - extrema (tuple): The (x0, alpha) around which to center the grid search.
    - data (pd.DataFrame): The observed data.
    - grid_size (int): The width/height of the grid (must be odd to have a center).
    - step_size (float): The step size between points in the grid.

    Returns:
    - tuple: (best_params, is_edge) where best_params is the best (x0, alpha) found and
             is_edge is a boolean indicating if this point is on the edge of the grid.
    """

    x0_center, alpha_center = extrema
    half_grid = grid_size // 2
    best_sse = float('inf')
    best_params = None
    is_edge = False

    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            x0 = x0_center + i * step_size
            alpha = alpha_center + j * step_size
            current_sse = sse((x0, alpha), data)
            
            if current_sse < best_sse:
                best_sse = current_sse
                best_params = (x0, alpha)
                is_edge = i in (-half_grid, half_grid) or j in (-half_grid, half_grid)

    return best_params, is_edge

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
    #print(partial_x0, partial_b)
    B = sp.groebner([partial_x0, partial_b], x0, b, order='lex')
    
    return B



#########
# 4D functions for Multi-dimensional work



def model_2D(t, A_1, alpha_1, A_2, alpha_2):
    """
    Calculates the sum of two exponentials at given times.
    
    Args:
        t (array-like): The time points for the calculation.
        A_1 (float): Amplitude of the first exponential term.
        alpha_1 (float): Decay rate of the first exponential term.
        A_2 (float): Amplitude of the second exponential term.
        alpha_2 (float): Decay rate of the second exponential term.
    
    Returns:
        np.array: The calculated sum of two exponential functions.
    """
    exp_input1 = float(alpha_1) * np.array(t, dtype=float)
    exp_input2 = float(alpha_2) * np.array(t, dtype=float)
    return A_1 * np.exp(exp_input1) + A_2 * np.exp(exp_input2)


def sse_2D(params, data):
   """
    Computes the sum of squared errors between model predictions and actual data.
    
    Args:
        params (tuple): A tuple of parameters (A_1, alpha_1, A_2, alpha_2) for the model.
        data (pd.DataFrame): DataFrame containing 'Time' and 'Data' columns.
    
    Returns:
        float: The calculated sum of squared errors.
    """
    A_1, alpha_1, A_2, alpha_2 = params
    predictions = model_2D(data['Time'], A_1, alpha_1, A_2, alpha_2)
    return np.sum((data['Data'] - predictions) ** 2)


def data_gen_2D(num_data_points=4, noise_level=0.1, A_1=1, alpha_1=1, A_2=1, alpha_2=1):
    """
    Generates synthetic data based on the sum of two exponentials plus noise.
    
    Args:
        num_data_points (int): Number of data points to generate.
        noise_level (float): Standard deviation of Gaussian noise.
        A_1 (float): Amplitude of the first exponential term.
        alpha_1 (float): Decay rate of the first exponential term.
        A_2 (float): Amplitude of the second exponential term.
        alpha_2 (float): Decay rate of the second exponential term.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'Time' and 'Data' containing the generated data.
    """
    t = np.arange(num_data_points)
    x = A_1 * np.exp(alpha_1 * t) + A_2 * np.exp(alpha_2 * t)
                                                 
    noise = noise_level * x *  np.random.normal(0, 1, size=t.shape) * 0.1
         
    x_noisy = x + noise    

    x_noisy_rational = np.array([min(sp.Rational(int(xn * 100), 100), 10000000) for xn in x_noisy])
    data = pd.DataFrame({'Time': t, 'Data': x_noisy_rational})
    return data

def groeb_2D(x_i, t_i):
    """
    Computes the Groebner basis for the system of equations derived from partial derivatives of SSE.
    
    Args:
        x_i (list): List of data points.
        t_i (list): List of time points corresponding to the data points.
    
    Returns:
        GroebnerBasis: The Groebner basis for the system, which simplifies solving the equations.
    """
    A_1, b_1, A_2, b_2 = sp.symbols('A_1 b_1 A_2 b_2')
    SSE_poly = sum([(x - (A_1 * b_1**t + A_2 * b_2**t))**2 for x, t in zip(x_i, t_i)])
    print(SSE_poly)
    partial_A_1 = sp.diff(SSE_poly, A_1)
    partial_b_1 = sp.diff(SSE_poly, b_1)
    partial_A_2 = sp.diff(SSE_poly, A_2)
    partial_b_2 = sp.diff(SSE_poly, b_2)
    
    B = sp.groebner([partial_A_1, partial_b_1, partial_A_2, partial_b_2], A_1, b_1, A_2, b_2, order='lex')
    
    return B


def evaluate_hessian_at_minimas_4x4(params, x_i, t_i):
    """
    Evaluates the Hessian matrix at the estimated minimas to ascertain their true nature,
    specifically determining if they are indeed minimas. This is achieved by constructing
    and analyzing a 4x4 Hessian matrix based on the second derivatives of the SSE polynomial,
    which is adjusted for a model involving combinations of parameters A_1, A_2, b_1, and b_2.

    Parameters:
    - params (list of tuples): Estimated parameter sets (A_1, A_2, b_1, b_2), each representing
      a unique combination of model parameters under consideration.
    - x_i (list): The observed data values.
    - t_i (list): The corresponding time values for the observed data.

    Returns:
    - list of tuples: Each tuple contains a parameter set and a boolean indicating if it represents
      a minimum, based on the positiveness of all eigenvalues of the Hessian matrix evaluated at those parameters.
    """
    # Define your symbols
    A_1, A_2, b_1, b_2 = sp.symbols('A_1 A_2 b_1 b_2')
    SSE_poly = sum([(x - A_1 * b_1 ** t + A_2 * b_2 ** t  )**2 for x, t in zip(x_i, t_i)])  # Adjust according to your model

    # Compute the second derivatives to form the Hessian matrix
    partials = [[A_1, A_2, b_1, b_2], [A_1, A_2, b_1, b_2]]
    Hessian = sp.Matrix([[sp.diff(sp.diff(SSE_poly, i), j) for i in partials[0]] for j in partials[1]])
    print(Hessian)
    minima_results = []
    for param in params:
        # Substitute parameter values into Hessian
        Hessian_at_point = Hessian.subs({A_1: param[0], A_2: param[1], b_1: param[2], b_2: param[3]})
        
        # Convert Hessian to a numerical matrix for eigenvalue computation
        Hessian_num = np.array(Hessian_at_point).astype(np.float64)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(Hessian_num)
        
        # Check if all eigenvalues are positive
        if all(val > 0 for val in eigenvalues):
            minima_results.append((param, True))
        else:
            minima_results.append((param, False))

    return minima_results