using HomotopyContinuation

# Define the variables
@var A_1 A_2 b_1 b_2

# Define the polynomial
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2
P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2

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

positive_b_solutions = filter(solution -> real(solution[3]) > 0 && real(solution[4]) > 0, practically_real_solutions)

# Check each solution
for solution in positive_b_solutions
    # Extract and print the real part of each solution component
    solution_vals = real.(solution)  # Convert solution to real parts
    println("Solution: ", solution_vals)
end