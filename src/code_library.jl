
using Gen
export exec, Yay, LeafType, TyRef, ObjRef, Primitive, SExpr, sexpr_node, sexpr_leaf, subexpressions, size, num_nodes, unwrap, new_env, call_func, CLibrary, CFunc, Env

mutable struct SExpr
    is_leaf::Bool
    leaf::Any #not SExprs here, should require smth about non listey s expr?? 
    children::Vector{SExpr}
    parent::Union{Tuple{SExpr,Int}, Nothing} # parent and which index of the child it is
end

struct Primitive
    name::Symbol
    arity::Int64
    #either a type of a list of types of length arity 
    input_type::Vector{DataType} #Replace these with functions, or maybe that's overkill, usually just checking type, but could check positive etc
    output_type::Type
end

struct LeafType
    type::Type
    distribution::Any #fix this, but Distribution breaks it 
    dist_args::Vector{Any}
end 

struct Yay
end 


function sexpr_node(children::Vector{SExpr}; parent=nothing)
    """outputs a node s expression"""
    expr = SExpr(false, nothing, children, parent)
    for (i,child) in enumerate(children)
        isnothing(child.parent) || error("arg already has parent")
        child.parent = (expr,i)
    end
    expr 
end

function sexpr_leaf(leaf; parent=nothing)
    """outputs a leaf s expression, like a number or something"""
    SExpr(true, leaf, Vector{SExpr}(), parent)
end

function Base.copy(e::SExpr)
    SExpr(
        e.is_leaf,
        e.leaf,
        [copy(child) for child in e.children],
        nothing,
    )
end

"child-first traversal"
function subexpressions(e::SExpr; subexprs = SExpr[])
    for child in e.children
        subexpressions(child, subexprs=subexprs)
    end
    push!(subexprs, e)
end

Base.size(e::SExpr) = if e.is_leaf 1. else sum(size, e.children, init=0.) end

num_nodes(e::SExpr) = 1 + sum(num_nodes, e.children, init=0)



Base.string(e::SExpr) = begin
    string_toret = ""
    if e.is_leaf
        if typeof(e.leaf) == Primitive
            string_toret = string_toret * string(e.leaf.name)
        else 
            string_toret = string_toret * string(e.leaf)
        end 
        
        @assert isempty(e.children)
    else
        string_toret = string_toret * "("
        
        for (i,child) in enumerate(e.children)
            #@show child, typeof(child)
            string_toret = string_toret * string(child)
            if i != length(e.children)
                string_toret = string_toret * " "
            end
        end
        string_toret = string_toret * ")"
        # print(io, "(", join(e.children, " "), ")")
    end
    return string_toret
end


Base.show(io::IO, e::SExpr) = begin    
    if e.is_leaf
        if typeof(e.leaf) == Primitive
            print(io, e.leaf.name)
        else 
            print(io, e.leaf)
        end 
        
        @assert isempty(e.children)
    else
        print(io, "(")
        for (i,child) in enumerate(e.children)
            Base.show(io, child)
            if i != length(e.children)
                print(io, " ")
            end
        end
        print(io, ")")
        # print(io, "(", join(e.children, " "), ")")
    end
end


function make_leaf(item)
    val = tryparse(Int, item)
    isnothing(val) || return sexpr_leaf(val)
    val = tryparse(Float64, item)
    isnothing(val) || return sexpr_leaf(val)
    return sexpr_leaf(Symbol(item))     
end

"""
Parse a string into an SExpr. Uses lisp-like syntax.

"foo" -> SAtom(:foo)
"(foo)" -> SList([SAtom(:foo)])
"((foo))" -> SList([SList([SAtom(:foo)])]) 
"(foo bar baz)" -> SList([SAtom(:foo), SAtom(:bar), SAtom(:baz)])
"()" -> SList([])

"""
function Base.parse(::Type{SExpr}, original_s::String)
    # add guaranteed parens around whole thing and guaranteed spacing around parens so they parse into their own items
    s = replace(original_s, "(" => " ( ", ")" => " ) ")

    # `split` will skip all quantities of all forms of whitespace
    items = split(s)
    length(items) > 2 || error("SExpr parse called on empty (or all whitespace) string")
    items[1] != ")" || error("SExpr starts with a closeparen. Found in $original_s")

    # this is a single symbol like "foo" or "bar"
    length(items) == 1 && return make_leaf(items[1])

    i=0
    expr_stack = SExpr[]
    # num_open_parens = Int[]

    while true
        i += 1

        i <= length(items) || error("unbalanced parens: unclosed parens in $original_s")

        if items[i] == "("
            push!(expr_stack, sexpr_node(SExpr[]))
        elseif items[i] == ")"
            # end an expression: pop the last SExpr off of expr_stack and add it to the SExpr at one level before that

            if length(expr_stack) == 1
                i == length(items) || error("trailing characters after final closeparen in $original_s")
                break
            end

            last = pop!(expr_stack)
            push!(expr_stack[end].children, last)
        else
            # any other item like "foo" or "+" is a symbol
            push!(expr_stack[end].children, make_leaf(items[i]))
        end
    end

    length(expr_stack) != 0 || error("unreachable - should have been caught by the first check for string emptiness")
    length(expr_stack) == 1 || error("unbalanced parens: not enough close parens in $original_s")

    return pop!(expr_stack)
end








# @auto_hash_equals struct HashNode
#     leaf::Union{Symbol,Nothing}
#     children::Vector{Int}
# end

# const global_struct_hash = Dict{HashNode, Int}()

# """
# sets structural hash value, possibly with side effects of updating the structural hash, and
# sets e.match.struct_hash. Requires .match to be set so we know this will be used immutably
# """
# function struct_hash(e::SExpr) :: Int
#     isnothing(e.match) || isnothing(e.match.struct_hash) || return e.match.struct_hash

#     node = HashNode(e.leaf, map(struct_hash,e.children))
#     if !haskey(global_struct_hash, node)
#         global_struct_hash[node] = length(global_struct_hash) + 1
#     end
#     isnothing(e.match) || (e.match.struct_hash = global_struct_hash[node])
#     return global_struct_hash[node]
# end

@inline function unwrap(e::SExpr)
    e.leaf
end

mutable struct CFunc
    body::Union{SExpr, Nothing}
    runs::Bool
end

mutable struct Ty
    step::CFunc
    attrs::Vector{Type}
end

# struct CLibrary
#     fns::Vector{CFunc}
# end

mutable struct State #stuff that changes across time 
    objs::Matrix{Float32}
    step_of_obj::Vector{Int} # which step function for each object
end

mutable struct ExecInfo
    constraints::ChoiceMap
    path::Vector{Any}
    has_constraints::Bool
end

mutable struct Env #stuff that doesn't change accross time
    # locals::Vector{Any}
    #state::State
    # step_of_obj::Vector{Int} # which step function for each object
    # sprites::Vector{Sprite}
    # code_library::Vector{CFunc}
    # exec::ExecInfo
end


# new_env() = Env([], Int[], Sprite[], CFunc[], ExecInfo(choicemap(), Symbol[], false))
#state State(Object[],[])

@gen function call_func(func_id, args, obj_id, state, code_library, einfo)
    # if !func.runs
    #     return nothing 
    # end 
    func = code_library[func_id]
    # save_locals = env.locals
    # env.locals = args
    # try
    res = {:body} ~ exec(func.body, args, obj_id, state, code_library, einfo, :body)
    # catch e
        #@show "CAUGHTEM"
        #@show e 
        # func.runs = false
        # return nothing 
    # end 
    # env.locals = save_locals
    return res
end

@gen function obj_dynamics(obj_id, state, code_library, einfo, with_constraints::ChoiceMap)
    # fn = env.code_library[env.step_of_obj[obj_id]]
    # args = Any[state.objs[obj_id]]

    empty!(einfo.path)
    einfo.constraints = with_constraints
    einfo.has_constraints = !isempty(with_constraints)

    # if fn.runs
        # try
    step ~ call_func(fn, args, obj_id, state, code_library, einfo);
        # catch e
            # fn.runs = false
        # end 
    # end 
    return nothing
end

# function constrain!(exec::ExecInfo, addr, val)
#     push!(exec.path, addr)
#     path = foldr(Pair,exec.path)
#     set_value!(exec.constraints, path, val)
#     pop!(exec.path);
# end

@gen function sample_or_constrained(einfo::ExecInfo, addr, fn, args)
    push!(einfo.path, addr)
    push!(einfo.path, :retval)
    path = foldr(Pair,einfo.path)
    res = if einfo.has_constraints
        get_value(einfo.constraints, path)
    else
        if !fn.runs
            return nothing
        end
        retval ~ fn(args...)
        @assert !has_value(einfo.constraints, path) || error("sampled same address twice")
        set_value!(einfo.constraints, path, retval)
        retval
    end
    pop!(einfo.path)
    pop!(einfo.path)
    return res
end


@gen function exec(e::SExpr, args, obj_id, state, code_library, einfo, addr)
    if e.is_leaf
        return e.leaf
    end

    @assert !(addr isa Pair)
    push!(einfo.path, addr)

    @assert length(e.children) > 0
    head = unwrap(e.children[1])

    ret = if head === :pass
        nothing
    elseif head === :load
        register = unwrap(e.children[2])
        state.objs[register, obj_id]
    elseif head === :arg
        i = unwrap(e.children[2])
        args[i]
    elseif head === :store
        register = unwrap(e.children[2])
        value ~ exec(e.children[3], args, obj_id, state, code_library, einfo, :value)
        # @show state.objs
        state.objs[register, obj_id] = value
        # @show state.objs
        # @show (register,obj_id,value)
        nothing
    # elseif head === :dt
    #     dt
    # elseif head === :elapsed
    #     elapsed
    elseif head === :call
        fn = unwrap(e.children[2])
        inner_args = Float64[]
        for i in 3:length(e.children)
            inner_arg = {(:args,i)} ~ exec(e.children[i], args, obj_id, state, code_library, einfo, (:args,i))
            push!(inner_args, inner_arg)
        end
        fn_res ~ call_func(fn, inner_args, obj_id, state, code_library, einfo)
        fn_res
    elseif head === :const
        unwrap(e.children[2])
    elseif head === :add
        a ~ exec(e.children[2], args, obj_id, state, code_library, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, code_library, einfo, :b)
        a + b
    elseif head === :mul
        a ~ exec(e.children[2], args, obj_id, state, code_library, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, code_library, einfo, :b)
        a * b
    elseif head === :normal
        mu ~ exec(e.children[2], args, obj_id, state, code_library, einfo, :mu)
        var ~ exec(e.children[3], args, obj_id, state, code_library, einfo, :var)
        ret_normal ~ sample_or_constrained(env.exec, :ret_normal, normal, [mu, var])
        ret_normal


    # elseif head === :spawn
    #     ty ~ exec(e.children[2], env, state, :ty)
    #     ty::TyRef
    #     sprite ~ exec(e.children[3], env, state, :sprite)
    #     pos ~ exec(e.children[4], env, state, :pos)
    #     attrs = []
    #     for (i,attr_ty) in enumerate(env.types[ty.id].attrs)
    #         attr = {(:attrs, i)} ~ exec(e.children[4+i], env, state, (:attrs,i))
    #         attr::attr_ty # type assert
    #         push!(attrs, attr)
    #     end
    #     obj = Object(sprite, pos, attrs)
    #     # todo have it reuse slots that have been freed
    #     push!(state.objs, obj)
    #     ObjRef(length(state.objs))
    # elseif head === :despawn
    #     obj ~ exec(e.children[2], env, state, :obj)
    #     obj::ObjRef
    #     obj.id = 0
    #     # find every objref to this and set it to null
    #     # for obj2 in env.state.objs
    #     #     if obj2 isa Object
    #     #         for i in eachindex(obj2.attrs)
    #     #             obj3 = obj2.attrs[i]
    #     #             if obj3 isa ObjRef && obj3.id == obj.id
    #     #                 obj2.attrs[i].id = 0
    #     #             end
    #     #         end
    #     #     end
    #     # end
    #     return nothing
    # elseif head === :ty
    #     id = unwrap(e.children[2])::Int
    #     @assert 0 < id <= length(env.types)
    #     TypeRef(id)
    # elseif head === :vec
    #     #@show e
    #     y ~ exec(e.children[2], env, state, :y)
    #     x ~ exec(e.children[3], env, state, :x)
    #     #@show y,x
    #     #@show Vec(y,x)
    #     Vec(y,x)
    # elseif head === :+
    #     a ~ exec(e.children[2], env, state, :a)
    #     b ~ exec(e.children[3], env, state, :b)
    #     a + b
    # elseif head === :get_local
    #     idx = unwrap(e.children[2])::Int
    #     env.locals[idx]
    # elseif head === :set_local
    #     idx = unwrap(e.children[2])::Int
    #     value ~ exec(e.children[3], env, state, :value)
    #     env.locals[idx] = value
    #     nothing
    # elseif head === :get_attr
    #     obj ~ exec(e.children[2], env, state, :obj)
    #     attr = unwrap(e.children[3])
    #     if attr isa Symbol
    #         getproperty(obj, attr)
    #     else
    #         obj.attrs[attr]
    #     end
    # elseif head === :set_attr
    #     obj ~ exec(e.children[2], env, state, :obj)
    #     attr = unwrap(e.children[3])
    #     value ~ exec(e.children[4], env, state, :value)
    #     if attr isa Symbol
    #         setproperty!(obj, attr, value)
    #     else
    #         obj.attrs[attr] = value
    #     end
    #     nothing
    # elseif head === :isnull
    #     obj ~ exec(e.children[2], env, state, :obj)
    #     obj::ObjRef
    #     obj.id == 0
    # elseif head === :null
    #     ObjRef(0)
    # # primitive distributions
    # elseif head === :normal_vec
    #     mu ~ exec(e.children[2], env, state, :mu)
    #     var ~ exec(e.children[3], env, state, :var)
    #     # ret_normal_vec ~ normal_vec(mu, var)
    #     # constrain!(env.exec, :ret_normal_vec, ret_normal_vec)
    #     ret_normal_vec ~ sample_or_constrained(env.exec, :ret_normal_vec, normal_vec, [mu, var])
    #     ret_normal_vec
    # elseif head === :normal
    #     mu ~ exec(e.children[2], env, state,  :mu)
    #     var ~ exec(e.children[3], env, state,  :var)
    #     # ret_normal ~ normal(mu, var)
    #     # constrain!(env.exec, :ret_normal, ret_normal)
    #     ret_normal ~ sample_or_constrained(env.exec, :ret_normal, normal, [mu, var])
    #     ret_normal
    # elseif head === :bernoulli
    #     p ~ exec(e.children[2], env, state,  :p)
    #     # ret_bernoulli ~ bernoulli(p)
    #     # constrain!(env.exec, :ret_bernoulli, ret_bernoulli)
    #     ret_bernoulli ~ sample_or_constrained(env.exec, :ret_bernoulli, bernoulli, [p])
    #     ret_bernoulli
    else
        @assert head isa Symbol "$(typeof(head))"
        @assert !startswith(string(head), ":") "the symbol $head has an extra leading colon (:) note that parsing sexprs inserts colons so you may have unnecessarily included one"
        error("unrecognized head $head ($(string(head))) $(head === :set_attr) $(head == :set_attr)")
    end

    println("$e -> $ret")

    @assert pop!(einfo.path) == addr
    return ret

end