

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


abstract type CStmt end
abstract type CExpr end

abstract type CDist <: CExpr end
abstract type CLiteral <: CExpr end

struct CArg
    # type and such
end

struct CFunc
    args::Vector{CArg}
    body::CStmt
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


@gen function eval_traced(expr::CExpr, env::Env) 

end

const eval_fns = Dict(
    CNormalVec => eval_traced,
)


# ctypeof(expr::CExprInner) = error("ctype_of not implemented for $(typeof(expr)))")
exec(expr::CExpr, env::Env) = error("eval not implemented for $(typeof(expr))")

# @gen function eval_traced(expr::CExpr, env::Env)

# end

function exec(func::CFunc, args::Vector{Any}, env::Env)
    @assert length(args) == length(func.args)
    env.locals = args
    exec(func.body, env)
end

function obj_init(obj::Object, env::Env)
    exec(env.code_library.fns[obj.init], Any[obj], env)
end

function obj_step(obj::Object, env::Env)
    exec(env.code_library.fns[obj.step], Any[obj], env)
end


# struct CType
#     base :: Symbol # :float :int :bool :obj :list :error :vec
#     obj_fields :: Union{Nothing, Set{Pair{Symbol, CType}}}
#     list_eltype :: Union{Nothing, CType}
# end

# struct CStmt
#     stmt::CStmtInner
# end

# struct CExpr
#     expr::CExprInner
    # type::CType

    # CExpr(expr::CExprInner) = new(expr, ctypeof(expr))
# end


# struct CLocalVar <: CVar
#     idx::Int
    # type::CType
# end

# struct CGlobalVar <: CVar
#     idx::Int
# end
# ctypeof(expr::CVar) = expr.type

struct CNormalVec <: CDist
    mu::CExpr
    var::CExpr
end
exec(expr::CNormalVec, env::Env) = normal_vec(exec(expr.mu, env), exec(expr.var, env))
# ctypeof(::CNormal) = :float

struct CFloat <: CLiteral
    data::Float64
end
exec(expr::CFloat, env::Env) = expr.data
# ctypeof(::CFloat) = :float


struct CGetLocal <: CExpr
    idx::Int
end
exec(expr::CGetLocal, env::Env) = env.locals[expr.idx]

struct CSetLocal <: CStmt
    idx::Int
    value::CExpr
end
exec(expr::CSetLocal, env::Env) = (env.locals[expr.idx] = exec(expr.value, env); nothing)

struct CGetGlobal <: CExpr
    idx::Int
end
exec(expr::CGetGlobal, env::Env) = env.globals[expr.idx]

struct CSetGlobal <: CStmt
    idx::Int
    value::CExpr
end
exec(expr::CSetGlobal, env::Env) = (env.globals[expr.idx] = exec(expr.value, env); nothing)

struct CGetAttr <: CExpr
    obj::CExpr
    attr::Int
    # type::CType
end
function exec(expr::CGetAttr, env::Env)
    obj = exec(expr.obj, env)
    if expr.attr == 0
        obj.pos
    else
        obj.attrs[expr.attr]
    end 
end
# ctypeof(expr::GetAttr) = expr.type


struct CSetAttr <: CStmt
    obj::CExpr
    attr::Int
    value::CExpr
end
function exec(expr::CSetAttr, env::Env)
    obj = exec(expr.obj, env)
    val = exec(expr.value, env);
    if expr.attr == 0
        obj.pos = val
    else
        obj.attrs[expr.attr] = val
    end
end

struct CPass <: CStmt end
exec(expr::CPass, env::Env) = nothing

export CFunc, CArg, CLibrary, Env, CNormalVec, CFloat, CGetLocal, CSetLocal, CGetGlobal, CSetGlobal, CGetAttr, CSetAttr, CPass
export obj_init, obj_step, new_env