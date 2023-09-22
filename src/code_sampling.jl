using Gen
import Distributions
export code_prior, new_production, choose_symbol

#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial

symbols = [:pos]
@dist choose_symbol() = symbols[categorical([1 / length(symbols) for i in 1:length(symbols)])]



#SAMPLING FUNCTIONS DISTRIBUTION 
struct UniformNode <: Gen.Distribution{Production} end
const uniform_node = UniformNode()
(::UniformNode)(output_type, must_be_leaf, productions) = random(UniformNode(), output_type, must_be_leaf, productions)


function Gen.random(::UniformNode, output_type, must_be_leaf, productions)
    """samples random function or leaftype with requirements"""
    return rand(possible_productions(output_type, must_be_leaf, productions))
end

function possible_productions(output_type, must_be_leaf, productions)
    """returns possible productions given output type and must_be_leaf"""
    prods = []

    for prod in productions
        if prod.ret_type === output_type && (!must_be_leaf || isempty(prod.arg_types))
            push!(prods, prod)
        end
    end

    return prods
end


function Gen.logpdf(::UniformNode, prod, output_type, must_be_leaf, productions)
    if prod.ret_type != output_type || (must_be_leaf && !isempty(prod.arg_types))
        return -Inf
    else
        return -log(length(possible_productions(output_type, must_be_leaf, productions)))
    end

end

#CODE SAMPLING 
@gen function code_prior(depth, output_type, productions)
    """samples code recursively!"""
    must_be_leaf = depth + 1 > 100

    node ~ uniform_node(output_type, must_be_leaf, productions)

    if node.val !== nothing
        return sexpr_leaf(node.val)
    elseif node.dist !== nothing
        (dist, args) = node.dist
        sampled_val ~ dist(args...)
        return sexpr_leaf(sampled_val)
    else
        child_vec = SExpr[sexpr_leaf(node.name)]
        for (i, arg_type) in enumerate(node.arg_types)
            sexpr = {(:sexpr, i)} ~ code_prior(depth + 1, arg_type, productions)
            push!(child_vec, sexpr)
        end
        return sexpr_node(child_vec)

    end
end

