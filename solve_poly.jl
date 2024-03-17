using HomotopyContinuation

# Define the variables
@var A_1 A_2 b_1 b_2

# Define the polynomial
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2

#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2 + (-A_1*b_1^6 - A_2*b_2^6 + 16137/20)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2 + (-A_1*b_1^6 - A_2*b_2^6 + 16137/20)^2 + (-A_1*b_1^7 - A_2*b_2^7 + 109663/50)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2 + (-A_1*b_1^6 - A_2*b_2^6 + 16137/20)^2 + (-A_1*b_1^7 - A_2*b_2^7 + 109663/50)^2 + (-A_1*b_1^8 - A_2*b_2^8 + 596191/100)^2
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2 + (-A_1*b_1^6 - A_2*b_2^6 + 16137/20)^2 + (-A_1*b_1^7 - A_2*b_2^7 + 109663/50)^2 + (-A_1*b_1^8 - A_2*b_2^8 + 596191/100)^2 + (-A_1*b_1^9 - A_2*b_2^9 + 10000)^2
#14
#P = (-A_1 - A_2 + 2)^2 + (-A_1*b_1 - A_2*b_2 + 543/100)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 1477/100)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4017/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 10919/100)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 14841/50)^2 + (-A_1*b_1^6 - A_2*b_2^6 + 16137/20)^2 + (-A_1*b_1^7 - A_2*b_2^7 + 109663/50)^2 + (-A_1*b_1^8 - A_2*b_2^8 + 596191/100)^2 + (-A_1*b_1^9 - A_2*b_2^9 + 405154/25)^2 + (-A_1*b_1^10 - A_2*b_2^10 + 4405293/100)^2 + (-A_1*b_1^11 - A_2*b_2^11 + 2993707/25)^2 + (-A_1*b_1^12 - A_2*b_2^12 + 16275479/50)^2 + (-A_1*b_1^13 - A_2*b_2^13 + 44241339/50)^2
#NOISY
P = (-A_1 - A_2 + 289/100)^2 + (-A_1*b_1 - A_2*b_2 + 123/50)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 127/10)^2 + (-A_1*b_1^3 - A_2*b_2^3 - 62/5)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 7799/50)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 54387/100)^2

#P = (-A_1 - A_2 + 201/100)^2 + (-A_1*b_1 - A_2*b_2 + 283/50)^2 + (-A_1*b_1^2 - A_2*b_2^2 + 661/50)^2 + (-A_1*b_1^3 - A_2*b_2^3 + 4879/100)^2 + (-A_1*b_1^4 - A_2*b_2^4 + 989/20)^2 + (-A_1*b_1^5 - A_2*b_2^5 + 9514/25)^2


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