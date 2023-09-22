using Gen
import Distributions
export code_prior, run_cp
#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial

symbols = [:pos]
@dist choose_symbol() = symbols[categorical([1 / length(symbols) for i in 1:length(symbols)])]

funcs = [
    Primitive(:pass, 0, [], Yay),
    Primitive(:vec, 2, [Float64, Float64], Vec),
    Primitive(:+, 2, [Float64, Float64], Float64),
    Primitive(:+, 2, [Vec, Vec], Vec),
    Primitive(:get_local, 1, [Int64], Object),
    Primitive(:get_attr, 2, [Object, Symbol], Vec), #todo make more general
    Primitive(:get_attr, 2, [Object, Int64], Float64), #todo make more general
    Primitive(:set_attr, 3, [Object, Symbol, Vec], Yay), #todo make more general 
]

leafs = [
    LeafType(Float64, Gen.normal, [0, 2]),
    LeafType(Int64, Gen.uniform_discrete, [1, 1]),
    LeafType(Symbol, choose_symbol, []),
]

#SAMPLING FUNCTIONS DISTRIBUTION 
struct UniformNode <: Gen.Distribution{Union{Primitive,LeafType}} end
const uniform_node = UniformNode()
(::UniformNode)(output_type, must_be_leaf) = random(UniformNode(), output_type, must_be_leaf)


function Gen.random(::UniformNode, output_type, must_be_leaf)
    """samples random function or leaftype with requirements"""
    return rand(possible_productions(output_type, must_be_leaf))
end

function possible_productions(output_type, must_be_leaf)
    """returns possible productions given output type and must_be_leaf"""
    possible_things = []

    #adding possible leafs 
    for leaf in leafs
        if leaf.type === output_type
            push!(possible_things, leaf)
        end
    end

    #adding funcs if not must_be_leaf
    if !must_be_leaf
        for func in funcs
            if func.output_type === output_type
                push!(possible_things, func)
            end
        end
    end

    return possible_things
end


function Gen.logpdf(::UniformNode, thing, output_type, must_be_leaf)
    if typeof(thing) == LeafType && thing.type != output_type
        return -Inf
    elseif typeof(thing) == Primitive && (must_be_leaf || thing.output_type != output_type)
        return -Inf
    else
        return -log(length(possible_productions(output_type, must_be_leaf)))
    end

end

#CODE SAMPLING 
@gen function code_prior(depth, output_type)
    """samples code recursively!"""
    must_be_leaf = depth + 1 > 10

    node ~ uniform_node(output_type, must_be_leaf)

    if typeof(node) == LeafType
        leaf ~ node.distribution(node.dist_args...)
        return sexpr_leaf(leaf)
    else
        child_vec = SExpr[sexpr_leaf(node.name)]
        for i in 1:node.arity
            sexpr = {(:sexpr, i)} ~ code_prior(depth + 1, node.input_type[i])
            push!(child_vec, sexpr)
        end
        return sexpr_node(child_vec)

    end
end



function run_cp(n=1, failable=false)
    for _ in 1:n
        if failable
            code = code_prior(0, Yay)
            println(code)
        else
            try
                code = code_prior(0, Yay)
                println(code)
            catch e
                println(e)
            end
        end
    end
end

