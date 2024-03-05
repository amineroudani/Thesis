import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import random 

from scipy.optimize import minimize
import sympy.plotting as spp
from sympy import symbols, diff, Poly, solve
from mpl_toolkits.mplot3d import Axes3D



# Defining Functions


def model(t, x0, alpha):
    """

    Defining the model.

    """
    # Cap the exponent to avoid overflow
    # capped_exponent = np.minimum(alpha * t, 1000)
    #print(f"Debug before exp: alpha={alpha}, t={t}, type(alpha)={type(alpha)}, type(t)={type(t)}, alpha*t={alpha*t}, type(alpha*t)={type(alpha*t)}")
    exp_input = float(alpha) * np.array(t, dtype=float)  # Ensure t is a numeric array
    return x0 * np.exp(exp_input)

def sse(params, data):
    """

    Defining the error function as SSE.
    
    params = tuple(x0, alpha)
    data = pd.DataFrame({'Time': , 'Data': })
   
    """
    x0, alpha = params
    #print(f"Debug in SSE: params={params}, type(params)={type(params)}")
    predictions = model(data['Time'], x0, alpha)
    x_data = data['Data']
    return np.sum((x_data - predictions) ** 2)


def data_gen(num_data_points=4, noise_level=0.1, alpha=2, x0=1):
    """
    
    Generating data based on x = x0 * exp(alpha * t) with fixed noise at noise_level in the interval [a, b] with a certain num_data_points

    num_data_points = int
    noise_level = float/int
    alpha = float/int
    x0 = float/int
    a = float/int
    b = float/int
    """
    t = np.array(list(range(num_data_points)))
    #print("Saved")
    x = x0 * np.exp(alpha * t)  # Using a fixed 'true' alpha=2 for data generation
    noise = np.random.normal(0, noise_level, x.shape)
    x_noisy = x + noise
    #Making x_noisy rational for Groebner basis computations later...
    x_noisy_rational = np.array([min(sp.Rational(int(xn * 100), 100), 10000) for xn in x_noisy])
    data = pd.DataFrame({'Time': t, 'Data': x_noisy_rational})
    return data



def run_optimization(num_runs, initial_guess, data, method='L-BFGS-B', bounds=None):
    """
    Running the optimization num_runs number of times with some guess initial_guess

    num_runs = int
    initial guess = tuple(x0, alpha)
    data = pd.DataFrame({'Time': , 'Data': })

    """
    # Running the optimization
    final_params = []
    for _ in range(num_runs):
        result = minimize(sse, initial_guess, data, method, bounds)
        
        # Check if the optimization was successful
        if result.success:
            final_params.append(result.x)
        else:
            # Handle unsuccessful optimization
            # For example, append None or log an error message
            final_params.append(None)

            print(f"Optimization failed: {result.message} for (x0, alpha) = {initial_guess}")

    # Filter out None values if there are any
    final_params = [param for param in final_params if param is not None]

    return np.array(final_params)


#Compute the error matrix using run_optimization and calculate_error.

def error_matrix(alpha_range, x0_range, data, num_runs=1):
    """
    Returns the error matrix giving the SSE error on the data for combinations alphas and x0s in the ranges alpha_range, x0_range based.

    alpha_range = list()
    x0_range = list()
    data = pd.DataFrame({'Time': , 'Data': })

    """
    # Initialize a matrix to store the errors
    error_matrix = np.ones((len(alpha_range), len(x0_range)))
    for i in range(len(error_matrix)):
        for j in range(len(error_matrix[0])):
            error_matrix[i][j] = 0
    # Fill in the matrix
    for i, alpha in enumerate(alpha_range):
        for j, x0 in enumerate(x0_range):
            initial_guess = (x0, alpha)
            final_param = run_optimization(num_runs, initial_guess, data)
            if len(final_param) > 0:
                # Assuming we only run the minimization once and it terminates.
                error_matrix[i, j] = sse(final_param[0], data)
            else:
                error_matrix[i, j] = np.inf
    return error_matrix



def sturm_sequence(p):
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
    sturm_seq = sturm_sequence(Poly(p, b))

    # Count sign changes at positive infinity and zero.
    sign_changes_at_inf = count_sign_changes(sturm_seq, 1e10)  # Simulate positive infinity.
    sign_changes_at_zero = count_sign_changes(sturm_seq, 0) #INCLUDING 0

    return sign_changes_at_zero - sign_changes_at_inf 

# Example usage:
#x = symbols('x')
#p = x**2 - 6*x # Define your polynomial here.
#num_positive_roots = count_positive_roots(B[1])
#print(f'Number of positive roots: {num_positive_roots}')



def evaluate_hessian_at_extremas(params, x_i, t_i):
    x0, b = sp.symbols('x0 b')
    # Define SSE_poly using the provided x_i and t_i
    SSE_poly = sum([(x - x0 * b**t)**2 for x, t in zip(x_i, t_i)])
    
    # Compute the general Hessian matrix
    partial_x0x0 = sp.diff(SSE_poly, x0, x0)
    partial_x0b = sp.diff(SSE_poly, x0, b)
    partial_bb = sp.diff(SSE_poly, b, b)
    
    hessian_general = sp.Matrix([
        [partial_x0x0, partial_x0b],
        [partial_x0b, partial_bb]  # Symmetry in mixed partials
    ])
    #print(f'Hessian General: {hessian_general}')    
    maxima_results = []

    for param in params:
        # Substitute the extremas into the general Hessian
        hessian_at_point = hessian_general.subs({x0: param[0], b: np.exp(param[1])})
        #print(f'Hessian at point: {hessian_at_point}')
        det = hessian_at_point.det()
        trace = hessian_at_point.trace()
        
        # Check for negative definiteness (indicative of a maximum) using det and trace
        if det > 0 and trace > 0:
            maxima_results.append((param, True))
        else:
            maxima_results.append((param, False))
    
    return maxima_results

# Example usage:
# params = [(root_x0, root_b), ...] # Replace with actual roots of the gradient
# x_i = [...] # Your x_i values
# t_i = [...] # Your t_i values
# maxima_checks = evaluate_hessian_at_extremas(params, x_i, t_i)

def find_x0_alpha_pairs(G,b_arr):
    first_eq = G[0]
    x0, b = sp.symbols('x0 b')
    
    x0_arr = []
    alpha_arr = []
    for b_i in b_arr:
        x0_i = sp.solve(first_eq.subs(b, b_i), x0)[0]
        x0_arr.append(x0_i)
        alpha = sp.log(b_i)
        alpha_arr.append(alpha)
    
    # Return the (x0, alpha) pairs
    ret = []
    for i in range(len(x0_arr)):
        ret.append((float(x0_arr[i]), float(alpha_arr[i])))
    return ret

def roots_symbolic(poly):
    x = symbols('b')
    # Define your polynomial
    # Solve polynomial
    roots = solve(poly, x)
    # Filter positive roots
    positive_roots_symbolic = [root.evalf() for root in roots if root.is_real and root > 0]
    return positive_roots_symbolic