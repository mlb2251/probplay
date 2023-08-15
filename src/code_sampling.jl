using Gen 
#export code_prior
#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial



abstract type Code end 
struct Leaf <: Code
    leafval::Float64
end
struct Plus <: Code
    a::Code
    b::Code
end


#what is spawn, type, isnull, null, 
#skipping get_local, set_local, get_attr, set_attr
#expr_types = [leaf, vec, +, normal_vec, normal, bernoulli]
expr_types = [Leaf, Plus]
@dist choose_expr_type() = expr_types[categorical([1/length(expr_types) for _ in expr_types])] #equally likely 



@gen function code_prior()
    expr_type ~ choose_expr_type()
    if expr_type == Leaf
        leafval ~ uniform(0, 1)
        return expr_type({:leafval} ~ uniform(0, 1))	
    elseif expr_type == Plus
        return expr_type({:a} ~ code_prior(), {:b} ~ code_prior())
    end 
    
end 

for _ in 1:3
    println(code_prior())
end 



# # Prior on kernels
# @gen function covariance_prior()
#     # Choose a type of kernel
#     kernel_type ~ choose_kernel_type()

#     # If this is a composite node, recursively generate subtrees
#     if in(kernel_type, [Plus, Times])
#         return kernel_type({:left} ~ covariance_prior(), {:right} ~ covariance_prior())
#     end
    
#     # Otherwise, generate parameters for the primitive kernel.
#     kernel_args = (kernel_type == Periodic) ? [{:scale} ~ uniform(0, 1), {:period} ~ uniform(0, 1)] : [{:param} ~ uniform(0, 1)]
#     return kernel_type(kernel_args...)
# end;