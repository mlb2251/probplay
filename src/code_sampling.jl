using Gen
import Distributions
export code_prior, new_production, choose_symbol

#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial

symbols = [:pos]
@dist choose_symbol() = symbols[categorical([1 / length(symbols) for i in 1:length(symbols)])]



#SAMPLING FUNCTIONS DISTRIBUTION 
# struct UniformProduction <: Gen.Distribution{Production} end
# const uniform_production = UniformProduction()
# (::UniformProduction)(output_type, dsl) = random(UniformProduction(), output_type, dsl)


# function Gen.random(::UniformProduction, output_type, dsl)
#     """samples random function or leaftype with requirements"""
#     return rand(dsl.productions_by_type[output_type])
# end

# function Gen.logpdf(::UniformProduction, prod, output_type, dsl)
#     if prod.ret_type != output_type
#         return -Inf
#     else
#         return -log(length(possible_productions(output_type, productions)))
#     end
# end


struct UniformSExpr <: Gen.Distribution{SExpr} end
const uniform_sexpr = UniformSExpr()
(::UniformSExpr)(output_type, max_depth, dsl) = random(UniformSExpr(), output_type, max_depth, dsl)

function Gen.random(::UniformSExpr, output_type, max_depth, dsl)

    max_depth <= 0 && return sexpr_leaf(:bottom; type=output_type)

    prods = dsl.productions_by_type[output_type]
    total_weight = sum(prods.weights)
    prod = labeled_cat(prods.productions, prods.weights ./ total_weight)

    if prod.val !== nothing
        return sexpr_leaf(prod.val; type=output_type)
    elseif prod.dist !== nothing
        (dist, args) = prod.dist
        sampled_val = dist(args...)
        return sexpr_leaf(sampled_val; type=output_type)
    else
        child_vec = SExpr[sexpr_leaf(prod.name; type=:fn)]
        for arg_type in prod.arg_types
            sexpr = uniform_sexpr(arg_type, max_depth-1, dsl)
            push!(child_vec, sexpr)
        end
        return sexpr_node(child_vec; type=output_type)
    end
end


function Gen.logpdf(::UniformSExpr, sexpr, type, max_depth, dsl)
    sexpr.type != type && return -Inf

    if max_depth <= 0
        sexpr.is_leaf && sexpr.leaf === :bottom && return 0.
        return -Inf
    end

    # figure out which production we are
    prods = dsl.productions_by_type[type]
    prod_idx = if sexpr.is_leaf
        findfirst(p -> p.name === prods.leaf_dist.name, prods.productions)
    else
        isempty(sexpr.children) && return -Inf
        head = sexpr.children[1]
        !head.is_leaf && return -Inf
        findfirst(p -> p.name === head.leaf, prods.productions)
    end
    prod_idx === nothing && return -Inf
    prod = prods.productions[prod_idx]

    log_weight = log(prods.weights[prod_idx]) - log(sum(prods.weights))
    # println("log_weight for $(prod.name) of type $type: $log_weight")

    if sexpr.is_leaf
        # we're a leaf!
        sexpr.leaf === :bottom && return -Inf
        if prod.val !== nothing
            sexpr.leaf === prod.val && return log_weight + 0. # logpdf of a constant has only one possible value so 0
            return -Inf
        else
            @assert prod.dist !== nothing "malformed production for leaf type - lacks both .const and .dist"
            (dist, args) = prod.dist
            lpdf = Gen.logpdf(dist, sexpr.leaf, args...)
            # println("lpdf of $(sexpr.leaf) in $(dist)($args): $lpdf")
            return log_weight + lpdf
        end
    end

    # not a leaf so recurse on children
    length(sexpr.children) - 1 != length(prod.arg_types) && return -Inf

    log_weight + sum(Gen.logpdf(uniform_sexpr, e, ty, max_depth-1, dsl) for (ty,e) in zip(prod.arg_types, sexpr.children[2:end]))
end


#CODE SAMPLING 
# @gen function code_prior(max_depth, output_type, productions)
#     """samples code recursively!"""

#     if max_depth <= 0
#         return sexpr_leaf(:bottom; type=output_type)
#     end

#     node ~ uniform_production(output_type, productions)

#     if node.val !== nothing
#         return sexpr_leaf(node.val; type=output_type)
#     elseif node.dist !== nothing
#         (dist, args) = node.dist
#         sampled_val ~ dist(args...)
#         return sexpr_leaf(sampled_val; type=output_type)
#     else
#         child_vec = SExpr[sexpr_leaf(node.name; type=:fn)]
#         for (i, arg_type) in enumerate(node.arg_types)
#             sexpr = {(:sexpr, i)} ~ code_prior(max_depth-1, arg_type, productions)
#             push!(child_vec, sexpr)
#         end
#         return sexpr_node(child_vec; type=output_type)

#     end
# end

