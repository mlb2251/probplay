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
    Primitive(:vec, 2, Float64, Float64), 
    Primitive(:+ , 2, Float64, Float64), 
    #get_local
    #set_local
    Primitive(:get_attr, 2, [Int64, Int64], Float64),#not sure about that one 
    #set_attr
    #isnull why would that ever happen 
    #null why ever 
    #normal vec 
    Primitive(:normal, 2, Float64, Float64),
    #bernoulli 
]
#arities = [2, 2, 2]
func_probs = [1/length(funcs) for _ in funcs]
#@dist choose_func_ind() = categorical(func_probs)
@dist choose_func() = funcs[categorical(func_probs)]



@gen function code_prior(depth, parent=nothing)
    depthp1 = depth + 1
    if depthp1 > 3
        is_leaf = true
    else
        is_leaf ~ choose_is_leaf()
    end

    if is_leaf
        leaf ~ uniform_discrete(0, 2) #can do better here
        
        return sexpr_leaf(leaf; parent) #might pass parent or not
    else
        #NOTE parent pointer will be dealt with in sexpr_node(...)
        func ~ choose_func()
        num_children = func.arity
        #vector of SExprs
        child_vec = SExpr[]
        #doing the first symbol (the function symbol)
        sexpr = sexpr_leaf(func)
        push!(child_vec, sexpr)

        #arguments 
        for child_ind in 1:num_children
            sexpr = {(:sexpr, child_ind)} ~ code_prior(depthp1) 
            push!(child_vec, sexpr)
        end 
        return sexpr_node(child_vec; parent)
    end 
end 




function run_cp(n=1)
    for _ in 1:n
        println(code_prior(0))
    end 
end 

