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
    return x0 * np.exp(alpha * t)

def sse(params, data):
    """

    Defining the error function as SSE.
    
    params = tuple(x0, alpha)
    data = pd.DataFrame({'Time': , 'Data': })
   
    """
    x0, alpha = params
    predictions = model(data['Time'], x0, alpha)
    x_data = data['Data']
    return np.sum((x_data - predictions) ** 2)


def data_gen(num_data_points, noise_level=0.1, alpha=2, x0=1, a=0, b=2):
    """
    
    Generating data based on x = x0 * exp(alpha * t) with fixed noise at noise_level in the interval [a, b] with a certain num_data_points

    num_data_points = int
    noise_level = float/int
    alpha = float/int
    x0 = float/int
    a = float/int
    b = float/int
    """
    t = np.linspace(a, b, num_data_points)
    x = x0 * np.exp(alpha * t)  # Using a fixed 'true' alpha=2 for data generation
    noise = np.random.normal(0, noise_level, x.shape)
    x_noisy = x + noise
    data = pd.DataFrame({'Time': t, 'Data': x_noisy})
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


