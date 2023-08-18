using Gen 
import Distributions
#export code_prior
#inspired by Gen Tutorial: https://www.gen.dev/tutorials/rj/tutorial



is_leaf_states = [true, false]
is_leaf_probs = [0.5, 0.5]
@dist choose_is_leaf() = is_leaf_states[categorical(is_leaf_probs)]

funcs = ["+", "normal"]#not sure if strings here is teh best 
arities = [2, 2]
func_probs = [0.5, 0.5]
@dist choose_func_ind() = categorical(func_probs)
#@dist choose_func_arr() = (funcs[categorical(func_probs)], arities[categorical(func_probs)])



@gen function code_prior(parent=nothing)
    is_leaf ~ choose_is_leaf()
    if is_leaf
        leaf ~ uniform_discrete(0, 1) #can do better here
        return sexpr_leaf(leaf; parent) #might pass parent or not
    else
        #NOTE parent pointer will be dealt with in sexpr_node(...)
        func_ind ~ choose_func_ind()
        (func, arity) = funcs[func_ind], arities[func_ind]
        num_children = arity
        #vector of SExprs
        child_vec = SExpr[]
        #doing the first symbol (the function symbol)
        sexpr = sexpr_leaf(func)
        push!(child_vec, sexpr)

        #arguments 
        for child_ind in 1:num_children
            sexpr = {(:sexpr, child_ind)} ~ code_prior() 
            push!(child_vec, sexpr)
        end 
        return sexpr_node(child_vec; parent)
    end 
end 




function run_cp(n=1)
    for _ in 1:n
        println(code_prior())
    end 
end 

