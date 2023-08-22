using Gen 
import Distributions
export code_prior, run_cp
#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial


# is_leaf_states = [true, false]
# is_leaf_probs = [0.5, 0.5]
# @dist choose_is_leaf() = is_leaf_states[categorical(is_leaf_probs)]

symbols = [Symbol("pos")]
@dist choose_symbol() = symbols[categorical([1/length(symbols) for i in 1:length(symbols)])]

funcs = [
    #spawn Primitive(:spawn, 3, [TyRef, Sprite, Vec], Nothing),#boolean is placeholder for idk
    #despawn todo
    #ty todo what even is this 
    Primitive(:pass, 0, [], Yay),
    Primitive(:vec, 2, [Float64, Float64], Vec), 
    Primitive(:+ , 2, [Float64, Float64], Float64), 
    Primitive(:+ , 2, [Vec, Vec], Vec),
    Primitive(:get_local, 1, [Int64], Object),
    #set_local
    Primitive(:get_attr, 2, [Object, Symbol], Vec), #todo make more general
    Primitive(:get_attr, 2, [Object, Int64], Float64), #todo make more general
    Primitive(:set_attr, 3, [Object, Symbol, Vec], Yay), #todo make more general 
    #isnull why would that ever happen 
    #null why ever 
    #normal vec 
    #Primitive(:normal, 2, [Float64, Float64], Float64),
    #bernoulli 
]

leafs = [

    LeafType(Float64, Gen.normal, [-1.9, 2]),
    #LeafType(Float64, Gen.poisson, [1]), #todo make more general
    LeafType(Int64, Gen.uniform_discrete, [1, 1]),
    LeafType(Symbol, choose_symbol, []),
]


#SAMPLING FUNCTIONS DISTRIBUTION 
struct Get_with_output <: Gen.Distribution{Union{Primitive, LeafType}} end 

function Gen.random(::Get_with_output, output_type, must_be_leaf)
    """samples random function or leaftype with requirements"""
    possible_things = []

    #adding possible leafs 
    for leaf in leafs
        if (output_type === nothing || leaf.type == output_type)
            push!(possible_things, leaf)
        end 
    end

    #adding funcs if not must_be_leaf
    if !must_be_leaf
        for func in funcs
            if (output_type === nothing || func.output_type == output_type)
                push!(possible_things, func)
            end 
        end 
    end

    #@show possible_things

    if length(possible_things) == 0
        if must_be_leaf
            return error("No leafs with type $output_type")
        else
            return error("No functions with output type $output_type")
        end 
    end 

    return possible_things[uniform_discrete(1, length(possible_things))]
    #possible_funcs[categorical([1/length(possible_funcs) for _ in 1:length(possible_funcs)])]#could weight this
end 

function get_num_things_with_output(output, must_be_leaf)
    """helper for log pdf
    counts number of funcs and leaftypes that have the requirements
    might be redundant with above function wasting time """
    num_things = 0

    #counting possible leafs
    for leaf in leafs
        if (output === nothing || leaf.type == output)
            num_things += 1
        end 
    end

    #counting possible funcs
    if !must_be_leaf
        for func in funcs
            if (output === nothing || func.output_type == output)
                num_things += 1
            end 
        end 
    end


    return num_things
end

function Gen.logpdf(::Get_with_output, thing, output_type, must_be_leaf)
    """likeliness"""

    if typeof(thing) == LeafType && thing.type != output_type
        return -Inf
    elseif typeof(thing) == Primitive && (must_be_leaf || thing.output_type != output_type)
        return -Inf
    else 
        return log(1/get_num_things_with_output(output_type, must_be_leaf))
    end

end

const get_with_output = Get_with_output()
(::Get_with_output)(output_type, must_be_leaf) = random(Get_with_output(), output_type, must_be_leaf)




#CODE SAMPLING 
@gen function code_prior(depth, output_type=nothing, parent=nothing)
    """samples code recursively!"""
    depthp1 = depth + 1
    if depthp1 > 10
        must_be_leaf = true
    else 
        must_be_leaf = false
    end

    thing ~ get_with_output(output_type, must_be_leaf)

    if typeof(thing) == LeafType
        leaf ~ thing.distribution(thing.dist_args...)
        return sexpr_leaf(leaf; parent)
    else #is type primitive 
        func = thing
        num_children = func.arity
        #vector of SExprs
        child_vec = SExpr[]
        #doing the first symbol (the function symbol)
        sexpr = sexpr_leaf(func)
        push!(child_vec, sexpr)
        #arguments 
        for child_ind in 1:num_children
            sexpr = {(:sexpr, child_ind)} ~ code_prior(depthp1, func.input_type[child_ind]) 
            push!(child_vec, sexpr)
        end 
        return sexpr_node(child_vec; parent)

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

