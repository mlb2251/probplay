using Gen
using LinearAlgebra
using Images
using AutoHashEquals
using Dates
import Distributions
using Revise #maybe? 


@auto_hash_equals struct Vec
    y::Float64
    x::Float64
end

Vec(y::Int, x::Int) = Vec(Float64(y), Float64(x))


Base.:+(v1::Vec, v2::Vec) = Vec(v1.y + v2.y, v1.x + v2.x)
Base.:-(v1::Vec, v2::Vec) = Vec(v1.y - v2.y, v1.x - v2.x)
Base.:*(a::Real, v::Vec) = Vec(a*v.y, a*v.x)
Base.:*(v::Vec, a::Real) = Vec(a*v.y, a*v.x)

Base.isapprox(v1::Vec, v2::Vec; kwargs...) = isapprox(v1.y, v2.y; kwargs...) && isapprox(v1.x, v2.x; kwargs...)

pixel_vec(v::Vec) = (floor(Int, v.y), floor(Int, v.x))

"""
given a vector of vectors, return a (min_vec, max_vec) pair
where min_vec is the vector of the minimum values of each component
"""
function min_max_vecs(vecs::Vector{Vec})
    ( 
        Vec(minimum(v -> v.y, vecs), minimum(v -> v.x, vecs)),
        Vec(maximum(v -> v.y, vecs), maximum(v -> v.x, vecs))
    )
end

"""
Rounds a Vec to be within (1,H+1-EPSILON) x (1,W+1-EPSILON)
"""
function inbounds_vec(v,H,W)
    Vec(min(max(1, v.y), H+1-EPSILON), min(max(1, v.x), W+1-EPSILON))
end

struct Yay
end 

struct Sprite
    mask::Matrix{Bool}
    color::Vector{Float64}
end

set_mask(sprite::Sprite, mask) = Sprite(mask, sprite.color)
set_color(sprite::Sprite, color) = Sprite(sprite.mask, color)


mutable struct ObjRef
    id::Int
end

struct TyRef
    id::Int
end

mutable struct Object
    # type :: Int
    sprite_index :: Int
    pos :: Vec
    attrs :: Vector{Any}

    Object(sprite_index, pos) = new(sprite_index, pos, [])
    Object(sprite_index, pos, attrs) = new(sprite_index, pos, attrs)
end

Base.copy(obj::Object) = Object(obj.sprite_index, obj.pos, copy(obj.float_attrs))

# Object(sprite_index, pos) = Object(sprite_index, pos, [], 0)

set_sprite(obj::Object, sprite_index) = Object(sprite_index, obj.pos, obj.attrs)
set_pos(obj::Object, pos) = Object(obj.sprite_index, pos, obj.attrs)

include("images.jl")

@dist labeled_cat(labels, probs) = labels[categorical(probs)]


const EPSILON = .001


struct UniformPosition <: Gen.Distribution{Vec} end
const uniform_position = UniformPosition()

function Gen.random(::UniformPosition, height, width)
    Vec(uniform(1., height+1-EPSILON), uniform(1., width+1-EPSILON))
end

function Gen.logpdf(::UniformPosition, pos, height, width)
    Gen.logpdf(uniform, pos.y, 1., height+1-EPSILON) + Gen.logpdf(uniform, pos.x, 1., width+1-EPSILON)
end




(::UniformPosition)(h, w) = random(UniformPosition(), h, w)

struct UniformDriftVec <: Gen.Distribution{Vec} end

function Gen.random(::UniformDriftVec, pos, max_drift)
    pos + Vec(rand(-max_drift:max_drift), rand(-max_drift:max_drift))
end

function Gen.logpdf(::UniformDriftVec, pos_new, pos, max_drift)
    # discrete uniform over square with side length 2*max_drift + 1
    return -2*log(2*max_drift + 1)
end

const uniform_drift_position = UniformDriftVec()

(::UniformDriftVec)(pos, max_drift) = random(UniformDriftVec(), pos, max_drift)

"""
normal distribution around a mu_vec with variance var
"""
struct NormalVec <: Gen.Distribution{Vec} end

function Gen.random(::NormalVec, mu_vec::Vec, var)
    Vec(normal(mu_vec.y,var), normal(mu_vec.x,var))
end

function Gen.logpdf(::NormalVec, v::Vec, mu_vec::Vec, var)
    Gen.logpdf(normal, v.y, mu_vec.y, var) + Gen.logpdf(normal, v.x, mu_vec.x, var)
end

const normal_vec = NormalVec()
(::NormalVec)(mu_vec, var) = random(NormalVec(), mu_vec, var)


struct Bernoulli2D <: Gen.Distribution{Array} end

function Gen.logpdf(::Bernoulli2D, x, p, h, w)
    num_values = prod(size(x))
    number_on = sum(x)
    return number_on * log(p) + (num_values - number_on) * log(1-p)
end

function Gen.random(::Bernoulli2D,  p, h, w)
    return rand(h,w) .< p
end

const bernoulli_2d = Bernoulli2D()

(::Bernoulli2D)(p, h, w) = random(Bernoulli2D(), p, h, w)

struct ImageLikelihood <: Gen.Distribution{Array} end

function Gen.logpdf(::ImageLikelihood, observed_image::Array{Float64,3}, rendered_image::Array{Float64,3}, var)
    # precomputing log(var) and assuming mu=0 both give speedups here
    log_var = log(var)
    sum(i -> - (@inbounds abs2((observed_image[i] - rendered_image[i]) / var) + log(2π)) / 2 - log_var, eachindex(observed_image))
end


function Gen.random(::ImageLikelihood, rendered_image, var)
    noise = rand(Distributions.Normal(0, var), size(rendered_image))
    # noise = mvnormal(zeros(size(rendered_image)), var * Maxxtrix(I, size(rendered_image)))
    rendered_image .+ noise
end

const image_likelihood = ImageLikelihood()
(::ImageLikelihood)(rendered_image, var) = random(ImageLikelihood(), rendered_image, var)

struct RGBDist <: Gen.Distribution{Vector{Float64}} end

function Gen.logpdf(::RGBDist, rgb)
    0. # uniform distribution over unit cube has density 1
end

function Gen.random(::RGBDist)
    rand(3)
end

const rgb_dist = RGBDist()

(::RGBDist)() = random(RGBDist())


function draw_region(objs, sprites, ymin, ymax, xmin, xmax)
    canvas = zeros(Float64, 3, ymax-ymin+1, xmax-xmin+1)
    draw_region(canvas, objs, sprites, ymin, ymax, xmin, xmax)
end

function draw_region(canvas, objs, sprites, ymin, ymax, xmin, xmax)
    for obj::Object in objs
        sprite_index = obj.sprite_index
        sprite = sprites[sprite_index]

        sprite_height, sprite_width = size(sprite.mask)

        x = floor(Int, obj.pos.x)
        y = floor(Int, obj.pos.y)

        # not at all in bounds
        if y > ymax || x > xmax || y+sprite_height-1 < ymin || x+sprite_width-1 < xmin
            continue
        end
        
        # starts where the object starts in the section, 1 if starts mid section and later if starts before the section 
        starti = max(1, ymin-y+1)
        startj = max(1, xmin-x+1)	
        stopi = min(sprite_height, ymax-y+1) 
        stopj = min(sprite_width, xmax-x+1)

        mask = @views sprite.mask[starti:stopi, startj:stopj]
        mask = reshape(mask, 1, size(mask)...)

        target = @views canvas[:, y+starti-ymin : y+stopi-ymin , x+startj-xmin : x+stopj-xmin]
        target .= ifelse.(mask, sprite.color, target)
    end
    canvas 
end 

draw(H, W, objs, sprites) = draw_region(objs, sprites, 1, H, 1, W)
draw(canvas, objs, sprites) = draw_region(canvas, objs, sprites, 1, size(canvas,2), 1, size(canvas,3))



include("code_library.jl")




# all_obj_dynamics = Map(obj_dynamics)


# Base.copy(state::State) = State([copy], copy(state.globals))

@gen function dynamics_and_render(t::Int, prev_state::State, env::Env, canvas_height, canvas_width, var)
    env.state = deepcopy(prev_state)
    for i in eachindex(env.state.objs)
        # @show i
        {:objs => i} ~ obj_dynamics(i, env, choicemap())
    end
    # for i in eachindex(env.state.objs)
        # pos = {:pos_noise => i} ~ normal_vec(env.state.objs[i].pos, 1.0)
        #sprite noise? 
        # env.state.objs[i].pos = pos
    # end

    rendered = draw(canvas_height, canvas_width, env.state.objs, env.sprites)
    observed_image ~ image_likelihood(rendered, var)
    return env.state
end

unfold_step = Unfold(dynamics_and_render)

@gen function make_attr(i)
    #can add more customization 
    attr ~ normal(0, 1)
    return attr 
end 

make_attrs = Map(make_attr)
 
@gen function make_object(i, H, W, num_sprites, env)
    sprite_index ~ uniform_discrete(1, num_sprites) 
    #pos ~ uniform_drift_position(Vec(0,0), 2) #never samping from this? why was using this not wrong?? 0.2? figure this out                                                                                      
    pos ~ uniform_position(H, W) 
    env.step_of_obj[i] = {:step_of_obj} ~ uniform_discrete(1, length(env.code_library))

    #velocity going here for now 
    #vel ~ normal(0, 1)


    #make it a normal attr 
    # attrs = [] 
    # for i in 1:1
    #     attr = {(:attr, i)} ~ make_attr()
    #     push!(attrs, attr)
    # end
    attrs ~ make_attrs(collect(1:1))

    #return Object(sprite_index, pos, [vel,])
    return Object(sprite_index, pos, attrs)
end

@gen function make_type(i, H, W) 
    width ~ uniform_discrete(1,W)
    height ~ uniform_discrete(1,H)
    shape ~ bernoulli_2d(0.5, height, width)
    color ~ rgb_dist()
    return Sprite(shape, color)
end

make_objects = Map(make_object)
make_sprites = Map(make_type)

# #testing
# testobjs = make_objects(collect(1:5), [10 for _ in 1:5], [20 for _ in 1:5])
# #@show testobjs
# testsprites = make_sprites(collect(1:4), [10 for _ in 1:4], [20 for _ in 1:4])
# #@show testsprites

@dist poisson_plus_1(lambda) = poisson(lambda) + 1

@gen function init_model(H,W,var)
    num_sprites ~ poisson_plus_1(4)
    env = new_env();
    

    env.sprites = {:init_sprites} ~ make_sprites(collect(1:num_sprites), [H for _ in 1:num_sprites], [W for _ in 1:num_sprites]) 

    sampled_code = {:sampled_code} ~ code_prior(0, Yay) #idk what to do here 
    @show sampled_code 

    env.code_library = [
        CFunc(parse(SExpr, string(sampled_code))),
        # move with local latent velocity
        # CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (normal_vec (get_attr (get_local 1) pos) 0.3) (get_attr (get_local 1) 1)))")),

        # random walk
        ####CFunc(parse(SExpr,"(set_attr (get_local 1) pos (normal_vec (get_attr (get_local 1) pos) 1.0))")),
        # stationary
        # CFunc(parse(SExpr,"(pass)")),
        # move const vel down right
        # CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (get_attr (get_local 1) pos) (vec 0.5 0.5)))")),
        # move const vel down
        # CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (get_attr (get_local 1) pos) (vec -2 0)))")),

        # #a little left
        # ##CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (get_attr (get_local 1) pos) (vec 0 -0.5)))")),
        # #perfect left 
        # CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (get_attr (get_local 1) pos) (vec 0 -2.3)))")),
        #  #move const vel right a little for frostbite
        # CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (get_attr (get_local 1) pos) (vec 0 5)))")),
        

        #goal get one const velocity func where velocity is a learned latent var pretty lit
        #add a list of attributes it needs
        ###CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (get_attr (get_local 1) pos) (vec 0 (get_attr (get_local 1) 1))))")),#velocity attribute is first
    ]

    #env.code_lib_reqs = [[], [1]] #addr 1 needed for the velocity code version 

    env.state = {:init_state} ~ init_state(H,W,num_sprites,env)

    rendered = draw(H, W, env.state.objs, env.sprites)
    {:observed_image} ~ image_likelihood(rendered, var)

    env
end

@gen function init_state(H,W,num_sprites,env)
    N ~ poisson(7)
    env.step_of_obj = fill(0,N) # will be set by make_objects
    init_objs ~  make_objects(collect(1:N), fill(H,N), fill(W,N), fill(num_sprites,N), fill(env,N))
    return State(init_objs, [])
end


# #testing
# testinitmodel = init_model(10, 20, 0.1)
# #@show testinitmodel

#  (static)
@gen (static) function model(H, W, T) 

    var = .1
    env = {:init} ~ init_model(H,W,var)
    #@show init_state
    {:steps} ~ unfold_step(T-1, env.state, env, H, W, var)

    return env
end

# #testing
# testmodel = model(10, 20, 8)
# @show testmodel


# end # module Model

