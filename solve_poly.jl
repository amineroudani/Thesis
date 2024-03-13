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
function evaluate_hessian_at_extremas(params, x_i, t_i, epsilon = 0.001)
    @variables x0 b
    SSE_poly = sum([(x - x0 * b^t)^2 for (x, t) in zip(x_i, t_i)])

    # Compute the second partial derivatives to construct the Hessian
    partial_x0x0 = Symbolics.derivative(SSE_poly, x0, x0)
    partial_x0b = Symbolics.derivative(SSE_poly, x0, b)
    partial_bb = Symbolics.derivative(SSE_poly, b, b)
    hessian_general = [partial_x0x0 partial_x0b; partial_x0b partial_bb]

    maxima_results = []
    for param in params
        # Use exp(param[2]) instead of param[2] directly if your parameters require exponentiation
        hessian_at_point = Symbolics.substitute(hessian_general, Dict(x0 => param[1], b => exp(param[3])))
        
        det = Symbolics.det(hessian_at_point)
        trace = Symbolics.tr(hessian_at_point)
        
        if det > 0 && trace > 0
            push!(maxima_results, (param, true))
        else
            push!(maxima_results, (param, false))
        end
    end
    
    return maxima_results
end

# Define the variables
@var A_1 A_2 b_1 b_2

# Define the polynomial
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2
P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2

# Compute the gradient
grad_P = [differentiate(P, var) for var in [A_1, A_2, b_1, b_2]]

# Solve the system formed by setting the gradient to zero
result = solve(grad_P)

# Epsilon for considering a solution as real
epsilon = 1e-2

# Get the solutions from the result
all_solutions = solutions(result)

# Filter for solutions that are "practically real" based on epsilon
practically_real_solutions = filter(solution -> all(val -> abs(imag(val)) < epsilon, solution), all_solutions)

println("Real solutions (considering a tolerance of $epsilon for imaginary parts):")
for solution in practically_real_solutions
    # Extract and print the real part of each solution component
    solution_vals = real.(solution)  # Convert solution to real parts, assuming it's already filtered for practical realness
    println("Solution: ", solution_vals)
end

