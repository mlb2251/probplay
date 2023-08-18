using Gen 
import Distributions
#export code_prior
#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial


is_leaf_states = [true, false]
is_leaf_probs = [0.5, 0.5]
@dist choose_is_leaf() = is_leaf_states[categorical(is_leaf_probs)]

#funcs = [+, Symbol("normal"), Symbol("vec")]#not sure if strings here is teh best 
funcs = [
    Primitive(:spawn, 3, [TyRef, Sprite, Vec], Nothing),#boolean is placeholder for idk
    #despawn todo
    #ty todo what even is this 
    Primitive(:vec, 2, [Float64, Float64], Float64), 
    Primitive(:+ , 2, [Float64, Float64], Float64), 
    #get_local
    #set_local
    Primitive(:get_attr, 2, [Int64, Int64], Float64),#not sure about that one 
    #set_attr
    #isnull why would that ever happen 
    #null why ever 
    #normal vec 
    Primitive(:normal, 2, [Float64, Float64], Float64),
    #bernoulli 
]
# #arities = [2, 2, 2]
# func_probs = [1/length(funcs) for _ in funcs]
# #@dist choose_func_ind() = categorical(func_probs)
# @dist choose_func(output_type) = funcs[categorical(func_probs)]


#SAMPLING FUNCTIONS DISTRIBUTION 
struct Func_with_output <: Gen.Distribution{Primitive} end 
const func_with_output = Func_with_output()
(::Func_with_output)(output_type) = random(func_with_output, output_type)

function Gen.random(::Func_with_output, output_type)
    possible_funcs = Primitive[]
    if output_type === nothing
        possible_funcs = funcs
    else 
        for func in funcs
            if func.output_type == output_type
                push!(possible_funcs, func)
            end 
        end
    end
    if length(possible_funcs) == 0
        return error("No functions with output type $output_type")
    end 

    return possible_funcs[uniform_discrete(1, length(possible_funcs))]
    #possible_funcs[categorical([1/length(possible_funcs) for _ in 1:length(possible_funcs)])]#could weight this
end 

function get_num_funcs_with_output(output)
    num_funcs = 0
    for func in funcs
        if func.output_type == output
            num_funcs += 1
        end 
    end 
    return num_funcs
end

function Gen.logpdf(::Func_with_output, func, output_type)
    if output_type === nothing
        return log(1/length(funcs))
    else
        if func.output_type != output_type
            return 0.0
        else
            return log(1/get_num_funcs_with_output(output_type))#could weight 
        end 
    end 
end


#CODE SAMPLING 
@gen function code_prior(depth, output_type=nothing, parent=nothing)
    depthp1 = depth + 1
    if depthp1 > 3 || output_type == Int64
        is_leaf = true
    else
        is_leaf ~ choose_is_leaf()
    end

    if is_leaf
        leaf ~ uniform_discrete(0, 2) #can do better here
        #ENFORCE OUTPUT TYPE 
        
        return sexpr_leaf(leaf; parent) #might pass parent or not
    else
        #NOTE parent pointer will be dealt with in sexpr_node(...)
        func ~ func_with_output(output_type)
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




function run_cp(n=1)
    for _ in 1:n
        try
            println(code_prior(0))
        catch e
            println(e)
        end
        #println(func_with_output(Float64))
    end 
end 

