
using Gen
export exec, SExpr, sexpr_node, sexpr_leaf, subexpressions, size, num_nodes, unwrap, new_env, call_func, CLibrary, CFunc, Env
"""

Types: Float, Int, Bool

Stmt ::=
    Pass
    Assign(Var(t), Expr(t))
    IfThenElse BExpr Stmt Stmt
    While BExpr Stmt
    Semi Stmt Stmt

Var(t) ::= LocalVar(t) || Attr(t) Symbol || GlobalVar(t)

Expr ::= FExpr || IExpr || BExpr

FExpr ::= FConst(arbitrary float) || FAdd FExpr FExpr || FMul FExpr FExpr || FDist || Var(Float)
IExpr ::= IConst(arbitrary int) || IAdd IExpr IExpr || IMul IExpr IExpr || IDist || Var(Int)
BExpr ::= True || False || Var(Bool)

FDist ::= Normal(FExpr,FExpr)

BExpr ::=
    Not BExpr
    Lt Expr Expr
    Eq Expr Expr

"""

mutable struct SExpr
    is_leaf::Bool
    leaf::Any
    children::Vector{SExpr}
    parent::Union{Tuple{SExpr,Int}, Nothing} # parent and which index of the child it is
end


function sexpr_node(children::Vector{SExpr}; parent=nothing)
    expr = SExpr(false, nothing, children, parent)
    for (i,child) in enumerate(children)
        isnothing(child.parent) || error("arg already has parent")
        child.parent = (expr,i)
    end
    expr 
end

function sexpr_leaf(leaf; parent=nothing)
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

Base.show(io::IO, e::SExpr) = begin    
    if e.is_leaf
        print(io, e.leaf)
        @assert isempty(e.children)
    else
        print(io, "(")
        for child in e.children
            Base.show(io, child)
            print(io, " ")
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
    body::SExpr
end

struct CLibrary
    fns::Vector{CFunc}
end

mutable struct Env
    locals::Vector{Any}
    globals::Vector{Any}
    objects::Vector{Object}
    sprites::Vector{Sprite}
    code_library::CLibrary
end

new_env() = Env([], [], Object[], Sprite[], CLibrary(CFunc[]))


@gen function call_func(func::CFunc, args::Vector{Any}, env::Env)
    save_locals = env.locals
    env.locals = args
    res ~ exec(func.body, env);
    env.locals = save_locals
    return res
end

@gen function exec(e::SExpr, env::Env) 
    if e.is_leaf
        return e.leaf
    end
    @assert length(e.children) > 0
    head = unwrap(e.children[1])

    if head === :pass
        return nothing
    elseif head === :normal_vec
        mu ~ exec(e.children[2], env)
        var ~ exec(e.children[3], env)
        res ~ normal_vec(mu, var)
        return res
    elseif head === :get_local
        idx = unwrap(e.children[2])::Int
        return env.locals[idx]
    elseif head === :set_local
        idx = unwrap(e.children[2])::Int
        value ~ exec(e.children[3], env)
        env.locals[idx] = value
        return nothing
    elseif head === :get_attr
        obj ~ exec(e.children[2], env)
        attr = unwrap(e.children[3])
        if attr isa Symbol
            return getproperty(obj, attr)
        else
            return obj.attrs[attr]
        end
    elseif head === :set_attr
        obj ~ exec(e.children[2], env)
        attr = unwrap(e.children[3])
        value ~ exec(e.children[4], env)
        if attr isa Symbol
            setproperty!(obj, attr, value)
        else
            obj.attrs[attr] = value
        end
        return nothing
    else
        @assert head isa Symbol "$(typeof(head))"
        @assert !startswith(string(head), ":") "the symbol $head has an extra leading colon (:) note that parsing sexprs inserts colons so you may have unnecessarily included one"
        error("unrecognized head $head ($(string(head))) $(head === :set_attr) $(head == :set_attr)")
    end
end