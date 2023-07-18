

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

struct Env
    locals::Vector{Any}
    globals::Vector{Any}
    code_library::CLibrary
end

@inline set_locals(env::Env, locals::Vector{Any}) = Env(locals, env.globals, env.code_library)

# ctypeof(expr::CExprInner) = error("ctype_of not implemented for $(typeof(expr)))")
eval(expr::CExpr, env::Env) = error("eval not implemented for $(typeof(expr))")

function eval(func::CFunc, args::Vector{Any}, env::Env)
    @assert length(args) == length(func.args)
    eval(func.body, set_locals(env,args))
end

function obj_init(obj::Object, env::Env)
    eval(env.code_library.fns[obj.init], Any[obj], env)
end

function obj_step(obj::Object, env::Env)
    eval(env.code_library.fns[obj.step], Any[obj], env)
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
eval(expr::CNormalVec, env::Env) = normal_vec(eval(expr.mu, env), eval(expr.var, env))
# ctypeof(::CNormal) = :float

struct CFloat <: CLiteral
    data::Float64
end
eval(expr::CFloat, env::Env) = expr.data
# ctypeof(::CFloat) = :float


struct CGetLocal <: CExpr
    idx::Int
end
eval(expr::CGetLocal, env::Env) = env.locals[expr.idx]

struct CSetLocal <: CStmt
    idx::Int
    value::CExpr
end
eval(expr::CSetLocal, env::Env) = (env.locals[expr.idx] = eval(expr.value, env); nothing)

struct CGetGlobal <: CExpr
    idx::Int
end
eval(expr::CGetGlobal, env::Env) = env.globals[expr.idx]

struct CSetGlobal <: CStmt
    idx::Int
    value::CExpr
end
eval(expr::CSetGlobal, env::Env) = (env.globals[expr.idx] = eval(expr.value, env); nothing)

struct CGetAttr <: CExpr
    obj::CExpr
    attr::Int
    # type::CType
end
function eval(expr::CGetAttr, env::Env)
    obj = eval(expr.obj, env)
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
function eval(expr::CSetAttr, env::Env)
    obj = eval(expr.obj, env)
    val = eval(expr.value, env);
    if expr.attr == 0
        obj.pos = val
    else
        obj.attrs[expr.attr] = val
    end
end

struct CPass <: CStmt end
eval(expr::CPass, env::Env) = nothing

export CFunc, CArg, CLibrary, Env, CNormalVec, CFloat, CGetLocal, CSetLocal, CGetGlobal, CSetGlobal, CGetAttr, CSetAttr, CPass
export obj_init, obj_step