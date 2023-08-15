using Gen 
#export code_prior
#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial



abstract type Code end 
struct Leaf <: Code
    leafval::Float64
end
struct Plus <: Code #is leaf true, leaf plus, children a b 
    a::Code
    b::Code
end


#what is spawn, type, isnull, null, 
#skipping get_local, set_local, get_attr, set_attr
#expr_types = [leaf, vec, +, normal_vec, normal, bernoulli]
expr_types = [Leaf, Plus]


function get_non_leaf_prob(depth, num_exprs, max_depth=3)
    return (1/num_exprs) * (max_depth - depth) / max_depth #latter term ranges from 1 to 0
end


#could use depth to skew towards leafs near the depth limit 
@dist function choose_expr_type(depth) 
    len = length(expr_types)
    @show len
    probs = [1 - get_non_leaf_prob(depth, len) * (len-1) ; [get_non_leaf_prob(depth, len) for _ in 1:(len-1)]]
    @show probs
    return expr_types[categorical(probs)] #equally likely 
end


@gen function code_prior(depth)
    depthp1 = depth + 1
    expr_type ~ choose_expr_type(depthp1)
    if expr_type == Leaf
        leafval ~ uniform(0, 1)
        return expr_type({:leafval} ~ uniform(0, 1))	
    elseif expr_type == Plus
        return expr_type({:a} ~ code_prior(depthp1), {:b} ~ code_prior(depthp1))
    end 
    
end 

for _ in 1:3
    println("code sample")
    println(code_prior(0))
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