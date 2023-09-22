
using Gen
export exec, Yay, LeafType, TyRef, ObjRef, Primitive, SExpr, sexpr_node, sexpr_leaf, subexpressions, size, num_nodes, unwrap, new_env, call_func, CLibrary, CFunc, Env, Library, add_fn, add_reg
export func_production, const_production, dist_production, uniform_sexpr

mutable struct SExpr
    is_leaf::Bool
    leaf::Any #not SExprs here, should require smth about non listey s expr?? 
    children::Vector{SExpr}
    # parent::Union{Tuple{SExpr,Int}, Nothing} # parent and which index of the child it is
end

struct Production
    name::Symbol
    arg_types::Vector{Symbol}
    ret_type::Symbol
    # for things that produce leaf values directly like constants
    val::Union{Nothing, Any}
    # for things that produce leaf values by sampling from a distribution
    dist::Union{Nothing, Tuple{Gen.Distribution, Vector{Any}}}
end

function func_production(name, arg_types, ret_type)
    Production(name, arg_types, ret_type, nothing, nothing)
end

function const_production(name, ret_type, val)
    Production(name, Symbol[], ret_type, val, nothing)
end

function dist_production(name, ret_type, dist, dist_args)
    Production(name, Symbol[], ret_type, nothing, (dist,dist_args))
end

function sexpr_node(children::Vector{SExpr})
    """outputs a node s expression"""
    SExpr(false, nothing, children)
end

function sexpr_leaf(leaf)
    """outputs a leaf s expression, like a number or something"""
    SExpr(true, leaf, Vector{SExpr}())
end

"""
Max arity of an expr. Assumes arguments are always of the form `(arg i)` where i>0.
For example (add (arg 1) 2.5) has arity 1.
"""
function arity(e::SExpr)
    arity = 0
    for subtree in subexpressions(e)
        # if subtree is of the form (arg i) and i is greater than the current max arity, update the max arity
        if !subtree.is_leaf && length(subtree.children) == 2 && subtree.children[1].leaf === :arg && subtree.children[2].leaf > arity
            arity = subtree.children[2].leaf
        end
    end
    arity
end

# function Base.copy(e::SExpr)
#     SExpr(
#         e.is_leaf,
#         e.leaf,
#         [copy(child) for child in e.children],
#         nothing,
#     )
# end

"child-first traversal"
function subexpressions(e::SExpr; subexprs=SExpr[])
    for child in e.children
        subexpressions(child, subexprs=subexprs)
    end
    push!(subexprs, e)
end

# function iter_subexprs(e::SExpr)
#     Iterators.flatten((Iterators.only(e), Iterators.map(iter_subexprs, e.children)))
# end
# Base.iterate(e::SExpr, state=1) = state > length(e.children) ? nothing : (e.children[state], state+1

Base.size(e::SExpr) =
    if e.is_leaf
        1.0
    else
        sum(size, e.children, init=0.0)
    end

num_nodes(e::SExpr) = 1 + sum(num_nodes, e.children, init=0)



# Base.string(e::SExpr) = begin
#     string_toret = ""
#     if e.is_leaf
#         if typeof(e.leaf) == Primitive
#             string_toret = string_toret * string(e.leaf.name)
#         else
#             string_toret = string_toret * string(e.leaf)
#         end

#         @assert isempty(e.children)
#     else
#         string_toret = string_toret * "("

#         for (i, child) in enumerate(e.children)
#             #@show child, typeof(child)
#             string_toret = string_toret * string(child)
#             if i != length(e.children)
#                 string_toret = string_toret * " "
#             end
#         end
#         string_toret = string_toret * ")"
#         # print(io, "(", join(e.children, " "), ")")
#     end
#     return string_toret
# end


Base.show(io::IO, e::SExpr) = begin
    if e.is_leaf
        print(io, e.leaf)
        @assert isempty(e.children)
    else
        print(io, "(")
        for (i, child) in enumerate(e.children)
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

    i = 0
    expr_stack = SExpr[]
    # num_open_parens = Int[]

    while true
        i += 1

        i <= length(items) || error("unbalanced parens: unclosed parens in $original_s")

        if items[i] == "("
            push!(expr_stack, sexpr_node(SExpr[]))
            # expr_stack[end].children[end].parent = expr_stack[end]
        elseif items[i] == ")"
            # end an expression: pop the last SExpr off of expr_stack and add it to the SExpr at one level before that

            if length(expr_stack) == 1
                i == length(items) || error("trailing characters after final closeparen in $original_s")
                break
            end

            last = pop!(expr_stack)
            push!(expr_stack[end].children, last)
            # expr_stack[end].children[end].parent = expr_stack[end]
        else
            # any other item like "foo" or "+" is a symbol
            push!(expr_stack[end].children, make_leaf(items[i]))
            # expr_stack[end].children[end].parent = expr_stack[end]
        end
    end

    length(expr_stack) != 0 || error("unreachable - should have been caught by the first check for string emptiness")
    length(expr_stack) == 1 || error("unbalanced parens: not enough close parens in $original_s")

    # for c in expr_stack[1].children
    #     c.parent = expr_stack[1]
    # end
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
    @assert e.is_leaf "$e is not a leaf"
    e.leaf
end

mutable struct CFunc
    body::Union{SExpr,Nothing}
    runs::Bool
end

CFunc(expr::SExpr) = CFunc(expr, true)
CFunc(s::String) = CFunc(parse(SExpr, s))


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


mutable struct Library
    fns::Vector{CFunc}
    abbreviations::Dict{Symbol,Int}

    register_of_name::Dict{Symbol,Int}
    name_of_register::Vector{Symbol}
    register_aliases::Dict{Symbol,Int}

    Library(num_registers) = new(Vector{CFunc}(), Dict{Symbol,Int}(), Dict{Symbol,Int}(), [Symbol("&$i") for i in 1:num_registers], Dict{Symbol,Int}())
end

function add_fn(lib::Library, func::CFunc, name=nothing)
    push!(lib.fns, func)
    if !isnothing(name)
        @assert !haskey(lib.abbreviations, name)
        lib.abbreviations[name] = length(lib.fns)
    end
end

add_fn(lib::Library, func::String, name=nothing) = add_fn(lib, CFunc(func), name)

function add_reg(lib::Library, name::Symbol, register::Int)
    lib.register_aliases[name] = register
end

get_fn(lib::Library, name::Symbol) = lib.fns[lib.abbreviations[name]]
get_fn(lib::Library, name::Int) = lib.fns[name]

get_register(lib::Library, name::Int) = name

function get_register(lib::Library, name::Symbol)
    if haskey(lib.register_of_name, name)
        return lib.register_of_name[name]
    elseif haskey(lib.register_aliases, name)
        return lib.register_aliases[name]
    else
        error("register $name not found")
    end
end


# new_env() = Env([], Int[], Sprite[], CFunc[], ExecInfo(choicemap(), Symbol[], false))
#state State(Object[],[])

@gen function call_func(func_id, args, obj_id, state, code_library, einfo)
    # if !func.runs
    #     return nothing 
    # end 
    if func_id isa Symbol
        @assert haskey(code_library.abbreviations, func_id) "function `$func_id` not found in code library"
        func_id = code_library.abbreviations[func_id]
    end

    func = code_library.fns[func_id]
    # save_locals = env.locals
    # env.locals = args
    # try

    if arity(func.body) != length(args)
        error("wrong number of arguments to function $func_id: $(func.body) invoked with args $args at trace position $(einfo.path)")
    end

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
    fn = state.step_of_obj[obj_id]
    args = []
    # args = Any[state.objs[obj_id]]

    empty!(einfo.path)
    einfo.constraints = with_constraints
    einfo.has_constraints = !isempty(with_constraints)

    # if fn.runs
    # try
    step ~ call_func(fn, args, obj_id, state, code_library, einfo)
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
    path = foldr(Pair, einfo.path)
    res = if einfo.has_constraints
        get_value(einfo.constraints, path)
    else
        # if !fn.runs
        #     return nothing
        # end
        retval ~ fn(args...)
        @assert !has_value(einfo.constraints, path) || error("sampled same address twice")
        set_value!(einfo.constraints, path, retval)
        retval
    end
    pop!(einfo.path)
    pop!(einfo.path)
    return res
end


@gen function exec(e::SExpr, args, obj_id, state, lib, einfo, addr)
    if e.is_leaf
        return e.leaf
    end

    # for c in e.children
    #     @assert c.parent === e "$e $c"
    # end

    # @show e
    # @show args

    @assert !(addr isa Pair)
    push!(einfo.path, addr)

    @assert length(e.children) > 0
    head = unwrap(e.children[1])

    ret = if head === :pass
        nothing
    elseif head === :load
        register ~ exec(e.children[2], args, obj_id, state, lib, einfo, :register)
        reg = get_register(lib, register)
        state.objs[reg, obj_id]
    elseif head === :arg
        i = unwrap(e.children[2])
        # println(join(einfo.path, " -> "))
        args[i]
    elseif head === :store
        register ~ exec(e.children[2], args, obj_id, state, lib, einfo, :register)
        reg = get_register(lib, register)
        value ~ exec(e.children[3], args, obj_id, state, lib, einfo, :value)
        # @show state.objs
        state.objs[reg, obj_id] = value
        # @show state.objs
        # @show (register,obj_id,value)
        nothing
        # elseif head === :dt
        #     dt
        # elseif head === :elapsed
        #     elapsed
    elseif head === :const
        unwrap(e.children[2])
    elseif head === :+
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        a + b
    elseif head === :*
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        a * b
    elseif head === :-
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        a - b
    elseif head === :<
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        a < b
    elseif head === :>
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        a > b
    elseif head === :(==)
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        a == b
    elseif head === :not
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        !a
    elseif head === :normal
        mu ~ exec(e.children[2], args, obj_id, state, lib, einfo, :mu)
        var ~ exec(e.children[3], args, obj_id, state, lib, einfo, :var)
        ret_normal ~ sample_or_constrained(einfo, :ret_normal, normal, [mu, var])
        ret_normal
    elseif head === :ifelse
        cond ~ exec(e.children[2], args, obj_id, state, lib, einfo, :cond)
        if cond
            branch ~ exec(e.children[3], args, obj_id, state, lib, einfo, :branch)
        else
            branch ~ exec(e.children[4], args, obj_id, state, lib, einfo, :branch)
        end
        branch
    elseif head === :seq
        a ~ exec(e.children[2], args, obj_id, state, lib, einfo, :a)
        b ~ exec(e.children[3], args, obj_id, state, lib, einfo, :b)
        b
    elseif head === :println
        for i in 2:length(e.children)
            a ~ exec(e.children[i], args, obj_id, state, lib, einfo, :a)
            print(a)
        end
        println()
        nothing
    elseif head === :str
        return string(e.children[2])

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

        # assume this is a function call
        fn ~ exec(e.children[1], args, obj_id, state, lib, einfo, :fn)
        inner_args = []
        for i in 2:length(e.children)
            inner_arg = {(:args, i-1)} ~ exec(e.children[i], args, obj_id, state, lib, einfo, (:args, i-1))
            push!(inner_args, inner_arg)
        end
        fn_res ~ call_func(fn, inner_args, obj_id, state, lib, einfo)
        fn_res

        # @assert head isa Symbol "$(typeof(head))"
        # @assert !startswith(string(head), ":") "the symbol $head has an extra leading colon (:) note that parsing sexprs inserts colons so you may have unnecessarily included one"
        # error("unrecognized head $head ($(string(head))) $(head === :set_attr) $(head == :set_attr)")
    end

    # println("$e -> $ret")

    @assert pop!(einfo.path) == addr
    return ret

end