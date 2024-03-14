using HomotopyContinuation
using Symbolics
using DataFrames, Random

# Model definition
function model_2D(t, A_1, alpha_1, A_2, alpha_2)
    exp_input1 = alpha_1 .* t
    exp_input2 = alpha_2 .* t
    return A_1 .* exp.(exp_input1) + A_2 .* exp.(exp_input2)
end

# SSE computation
function sse_2D(params, data)
    A_1, alpha_1, A_2, alpha_2 = params
    predictions = model_2D(data.Time, A_1, alpha_1, A_2, alpha_2)
    return sum((data.Data - predictions) .^ 2)
end

# Data generation
function data_gen_2D(num_data_points=4, noise_level=0.1, A_1=1, alpha_1=1, A_2=1, alpha_2=1)
    t = 0:(num_data_points-1)
    x = model_2D(t, A_1, alpha_1, A_2, alpha_2)
    
    noise = noise_level .* x .* randn(length(t)) .* 0.1
    x_noisy = x + noise
    
    x_noisy_rational = [min(rationalize(xn * 100) / 100, 10000) for xn in x_noisy]
    data = DataFrame(Time = t, Data = x_noisy_rational)
    return data
end

# Hessian 
function evaluate_hessian_at_extremas(params, x_i, t_i, A_1_val, A_2_val, b_1_val, b_2_val, epsilon = 0.001)
    @variables A_1 A_2 b_1 b_2
    SSE_poly = sum([(x - (A_1 * b_1^t + A_2 * b_2^t))^2 for (x, t) in zip(x_i, t_i)])
    
    # Compute the second partial derivatives to construct the Hessian
    H = Symbolics.hessian(SSE_poly, [A_1, A_2, b_1, b_2])

    minima_results = []
    for param in params
        # Substitute the parameter values into the Hessian matrix
        H_sub = Symbolics.substitute(H, Dict(A_1 => A_1_val, A_2 => A_2_val, b_1 => b_1_val, b_2 => b_2_val))
        
        # Evaluate the determinant and the trace of the Hessian matrix
        det = Symbolics.det(H_sub)
        trace = Symbolics.tr(H_sub)
        
        # Check the conditions for a minimum: det > 0 indicates a local extremum; trace > 0 combined with det > 0 indicates a local minimum
        if det > 0 && all(eig -> eig > 0, Symbolics.eigenvalues(H_sub))
            push!(minima_results, (param, true))
        else
            push!(minima_results, (param, false))
        end
    end
    
    return minima_results
end

# SSE_poly
function compute_SSE_poly(x_i, t_i, A_1, A_2, b_1, b_2)
    @variables A1 A2 b1 b2
    SSE_poly = sum([(x - (A1 * b1^t + A2 * b2^t))^2 for (x, t) in zip(x_i, t_i)])
    return SSE_poly
end

# Main loop varying model parameters and performing calculations
model_parameters = [(1, 1, 0.5, 0.5), (1.05, 0.95, 0.55, 0.45), (0.95, 1.05, 0.45, 0.55)]
epsilon = 1e-2 # Tolerance for considering a solution as real
num_data_points = 4 # Number of data points to generate

for (A_1, alpha_1, A_2, alpha_2) in model_parameters
    # Generate data with the current set of parameters
    data = data_gen_2D(num_data_points, 0, A_1, alpha_1, A_2, alpha_2)
    
    # Extract x_i and t_i from generated data
    x_i = data.Data
    t_i = data.Time
    
    # Compute SSE polynomial
    SSE_poly = compute_SSE_poly(x_i, t_i, A_1, A_2, exp(alpha_1), exp(alpha_2))
    
    # Solve for roots using homotopic continuation, ensuring b1, b2 > 0
    @variables A_1 A_2 b_1 b_2
    grad_P = [Symbolics.derivative(SSE_poly, var) for var in [A_1, A_2, b_1, b_2]]
    result = solve(grad_P)
    all_solutions = solutions(result)
    
    # Filter for real solutions and ensure b_1, b_2 > 0
    real_solutions = filter(solution -> all(val -> abs(imag(val)) < epsilon && real(val) > 0, solution[b_1:b_2]), all_solutions)
    
    # Evaluate Hessian at real solutions to identify minima
    minima_info = evaluate_hessian_at_extremas(real_solutions, x_i, t_i)
    
    println("For parameters A_1 = $A_1, A_2 = $A_2, alpha_1 = $alpha_1, alpha_2 = $alpha_2")
    println("Number of minima found: ", count(((sol, is_minima)) -> is_minima, minima_info))
    println("Minima information: ", minima_info)
end
