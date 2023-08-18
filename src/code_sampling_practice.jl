using Gen 
import Distributions
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
struct Normal <: Code
    mu::Code
    sigma::Code
end 


#what is spawn, type, isnull, null, 
#skipping get_local, set_local, get_attr, set_attr
#expr_types = [leaf, vec, +, normal_vec, normal, bernoulli]
expr_types = [Leaf, Plus, Normal]
TYPES_LEN = length(expr_types)
#expr_probabilities = [1/TYPES_LEN for _ in 1:TYPES_LEN]
expr_probabilities = [0.5, 0.25, 0.25]
@dist choose_expr_type() = expr_types[categorical(expr_probabilities)]



# function get_non_leaf_prob(depthh, num_exprs=TYPES_LEN, max_depthh=3)
#     return (1/num_exprs) * (max_depthh - depthh) / max_depthh #latter term ranges from 1 to 0
# end
# bla = get_non_leaf_prob(0)
# @dist choose_expr_type(non_leaf_prob) = expr_types[categorical([non_leaf_prob ; [0.5 for _ in 1:length(expr_types)-1]])];
# #I see why the random dist vars cant work here ug 



@gen function code_prior(depthh)
    depthhp1 = depthh + 1 #not used rn
    expr_type ~ choose_expr_type()
    if expr_type == Leaf
        leafval ~ uniform(0, 1)
        return expr_type({:leafval} ~ uniform(0, 1))	
    elseif expr_type == Plus
        return expr_type({:a} ~ code_prior(depthhp1), {:b} ~ code_prior(depthhp1))
    elseif expr_type == Normal
        return expr_type({:mu} ~ code_prior(depthhp1), {:sigma} ~code_prior(depthhp1))
    end 
    
end 

# for _ in 1:3
#     println("code sample")
#     println(code_prior(0))
# end 

function run_cp(n)
    for _ in 1:n
        println(code_prior(0))
    end 
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