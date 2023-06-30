using Gen
using LinearAlgebra
using Images
using Distributions
using Plots
using AutoHashEquals
using Dates

@auto_hash_equals struct Position
    y::Int
    x::Int
end

mutable struct Sprite_Type
    mask::Matrix{Bool}
    color::Vector{Float64}
end

struct Object
    sprite_index :: Int  
    pos :: Position
end 

include("images.jl")

@dist labeled_cat(labels, probs) = labels[categorical(probs)]

struct UniformPosition <: Gen.Distribution{Position} end

function Gen.random(::UniformPosition, height, width)
    Position(rand(1:height), rand(1:width))
end

function Gen.logpdf(::UniformPosition, pos, height, width)
    if !(0 < pos.y <= height && 0 < pos.x <= width)
        return -Inf
    else
        # uniform distribution over height*width positions
        return -log(height*width)
    end
end

const uniform_position = UniformPosition()

(::UniformPosition)(h, w) = random(UniformPosition(), h, w)


struct UniformDriftPosition <: Gen.Distribution{Position} end

function Gen.random(::UniformDriftPosition, pos, max_drift)
    Position(pos.y + rand(-max_drift:max_drift),
             pos.x + rand(-max_drift:max_drift))
end

function Gen.logpdf(::UniformDriftPosition, pos_new, pos, max_drift)
    # discrete uniform over square with side length 2*max_drift + 1
    return -2*log(2*max_drift + 1)
end

const uniform_drift_position = UniformDriftPosition()

(::UniformDriftPosition)(pos, max_drift) = random(UniformDriftPosition(), pos, max_drift)



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

# @assert isapprox(Gen.logpdf(M.image_likelihood, vals, vals2, .1), Gen.logpdf(broadcasted_normal, vals - vals2, zeros(Float64,size(vals)), .1))


function Gen.random(::ImageLikelihood, rendered_image, var)
    noise = rand(Normal(0, var), size(rendered_image))
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


function canvas(height=210, width=160)
    zeros(Float64, 3, height, width)
end


function draw(H, W, objs, sprites)
    canvas = zeros(Float64, 3, H, W)

    for obj::Object in objs 
        sprite_index = obj.sprite_index
        sprite_type = sprites[sprite_index]

        sprite_height, sprite_width = size(sprite_type.mask)

        for i in 1:sprite_height, j in 1:sprite_width
            if sprite_type.mask[i,j]
                offy = obj.pos.y+i-1
                offx = obj.pos.x+j-1
                if 0 < offy <= size(canvas,2) && 0 < offx <= size(canvas,3)
                    @inbounds canvas[:, offy,offx] = sprite_type.color
                end
            end
        end
    end
    canvas
end


function sim(T)
    (trace, _) = generate(model, (100, 100, T))
    return trace
end


module Model
using Gen
import ..Position, ..Sprite_Type, ..Object, ..draw, ..image_likelihood, ..bernoulli_2d, ..rgb_dist, ..uniform_position, ..uniform_drift_position

@gen (static) function obj_dynamics(obj::Object)
    pos ~ uniform_drift_position(obj.pos,2);
    return Object(obj.sprite_index, pos)
end

all_obj_dynamics = Map(obj_dynamics)

struct State
    objs::Vector{Object}
    sprites::Vector{Sprite_Type}
end

@gen (static) function dynamics_and_render(t::Int, prev_state::State, canvas_height, canvas_width, var)
    objs ~ all_obj_dynamics(prev_state.objs)
    sprites = prev_state.sprites 
    rendered = draw(canvas_height, canvas_width, objs, sprites)
    observed_image ~ image_likelihood(rendered, var)
    return State(objs, sprites)
end

unfold_step = Unfold(dynamics_and_render)


 
@gen (static) function make_object(i, H, W)
    NUM_SPRITE_TYPES = 4
    
    sprite_index ~ uniform_discrete(1, NUM_SPRITE_TYPES) 
    #pos ~ uniform_drift_position(Position(0,0), 2) #never samping from this? why was using this not wrong?? 0.2? figure this out                                                                                      
    pos ~ uniform_position(H, W) 

    return Object(sprite_index, pos)
end

@gen (static) function make_type(i, H, W) 
    width ~ uniform_discrete(1,W)
    height ~ uniform_discrete(1,H)
    shape ~ bernoulli_2d(0.5, height, width)
    color ~ rgb_dist()
    return Sprite_Type(shape, color)
end

make_objects = Map(make_object)
make_sprites = Map(make_type)

# #testing
# testobjs = make_objects(collect(1:5), [10 for _ in 1:5], [20 for _ in 1:5])
# #@show testobjs
# testsprites = make_sprites(collect(1:4), [10 for _ in 1:4], [20 for _ in 1:4])
# #@show testsprites


@gen function init_model(H,W,var)
    NUM_SPRITE_TYPES = 4
    N ~ poisson(7)
    objs = {:init_objs} ~  make_objects(collect(1:N), [H for _ in 1:N], [W for _ in 1:N])
    sprites = {:init_sprites} ~ make_sprites(collect(1:NUM_SPRITE_TYPES), [H for _ in 1:NUM_SPRITE_TYPES], [W for _ in 1:NUM_SPRITE_TYPES]) 

    rendered = draw(H, W, objs, sprites)
    {:observed_image} ~ image_likelihood(rendered, var)

    State(objs, sprites) 
end

# #testing
# testinitmodel = init_model(10, 20, 0.1)
# #@show testinitmodel


@gen (static) function model(H, W, T) 

    var = .1
    init_state = {:init} ~ init_model(H,W,var)
    #@show init_state
    state = {:steps} ~ unfold_step(T-1, init_state, H, W, var)

    return state
end

# #testing
# testmodel = model(10, 20, 8)
# @show testmodel


end # module Model

