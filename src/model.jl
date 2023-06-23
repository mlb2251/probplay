

module M

using Gen
using LinearAlgebra
using Images
using Distributions
using Plots
using AutoHashEquals
# using OffsetArrays
using Dates
# import FunctionalCollections: PersistentVector

# include("html.jl")

@auto_hash_equals struct Position
    y::Int
    x::Int
end

struct Sprite
    mask::Matrix{Bool}
    color::Vector{Float64}
end

struct Object
    sprite::Sprite
    pos::Position
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

# canvas = zeros(3,100,100)

# pos_x = 1
# pos_y = 1
# mask = reshape(rand(30,30) .< 0.5, (1,30,30))
# values = rand(3, sum(mask))


# full_image_mask = falses(size(canvas))
# full_image_mask[:,pos_x: pos_x+size(mask)[2]-1, pos_y: pos_y+size(mask)[3]-1] = mask
# canvas[:,full_image_mask] = values


function draw(H, W, objs)
    # max_sprite_height = maximum([size(obj.sprite.mask)[1] for obj::Object in objs])
    # max_sprite_width = maximum([size(obj.sprite.mask)[2] for obj::Object in objs])
    
    # hpad = 20
    # wpad = 20
    # canvas = zeros(Float64, 3, H + hpad * 2, W + wpad * 2)
    canvas = zeros(Float64, 3, H, W)

    for obj::Object in objs # this type annotation is a 10x speedup :0
        sprite = obj.sprite

        sprite_height, sprite_width = size(sprite.mask)
        
        # offset_sprite = OffsetArray(sprite.mask, obj.pos.x, obj.pos.y)

        # @show paddedviews(0., canvas, offset_sprite)

        # color = reinterpret(Tuple{Float64,Float64,Float64}, sprite.color)
        # color = (sprite.color[1], sprite.color[2], sprite.color[3])



        # mask = BitArray(reshape(sprite.mask, (1,sprite_height,sprite_width)))
        # mask = BitArray(repeat(reshape(sprite.mask, (1,sprite_height,sprite_width)), outer = (3,1,1)))
        # mask = reshape(sprite.mask, (1,sprite_height,sprite_width))

        # @show size(mask)
        # @show size(canvas[:, hpad+obj.pos.y:obj.pos.y + sprite_height - 1, wpad+obj.pos.x:obj.pos.x + sprite_width - 1])

        # @show typeof(sprite.mask)

        # @show size(canvas[:, hpad+obj.pos.y:hpad+obj.pos.y + sprite_height - 1, wpad+obj.pos.x:wpad+obj.pos.x + sprite_width - 1])
        # @show size(canvas[:, hpad+obj.pos.y:hpad+obj.pos.y + sprite_height - 1, wpad+obj.pos.x:wpad+obj.pos.x + sprite_width - 1][mask])
        # @show size(mask)

        # canvas[:, hpad+obj.pos.y:hpad+obj.pos.y + sprite_height - 1, wpad+obj.pos.x:wpad+obj.pos.x + sprite_width - 1][mask] .= sprite.color

        # canvas[:, hpad+obj.pos.y:hpad+obj.pos.y + sprite_height - 1, wpad+obj.pos.x:wpad+obj.pos.x + sprite_width - 1] .= 

        # color_sprite = sprite.color .* mask

        # z = @view canvas[:, hpad+obj.pos.y:hpad+obj.pos.y + sprite_height - 1, wpad+obj.pos.x:wpad+obj.pos.x + sprite_width - 1]
        # n = size(z[.!mask])

        # @show size(z[mask])

        # z .*= .!mask
        # z .+= color_sprite

        # z[mask] .= sprite.color
        # z[mask] .= rand()
        # if n[1] > 1000000000000

        #     println("hi")
        # end

        for i in 1:sprite_height, j in 1:sprite_width
            if sprite.mask[i,j]
                offy = obj.pos.y+i-1
                offx = obj.pos.x+j-1
                if 0 < offy <= size(canvas,2) && 0 < offx <= size(canvas,3)
                    @inbounds canvas[:, offy,offx] = sprite.color
                end
            end
        end
    end
    # canvas[:,hpad:end-hpad, wpad:end-wpad]
    # Array(reinterpret(reshape, Float64, canvas))
    canvas
end

@gen (static) function obj_dynamics(obj::Object)
    pos ~ uniform_drift_position(obj.pos,2);
    return Object(obj.sprite, pos)
end

all_obj_dynamics = Map(obj_dynamics)

struct State
    objs::Vector{Object}
end

@gen (static) function dynamics_and_render(t::Int, prev_state::State, canvas_height, canvas_width, var)
    objs ~ all_obj_dynamics(prev_state.objs)
    rendered = draw(canvas_height, canvas_width, objs)
    observed_image ~ image_likelihood(rendered, var)
    return State(objs)
end

unfold_step = Unfold(dynamics_and_render)

@gen (static) function make_object(i, H, W)
    width = ~ uniform_discrete(1,W)
    height = ~ uniform_discrete(1,H)
    shape ~ bernoulli_2d(0.5, height, width)
    color ~ rgb_dist()
    pos ~ uniform_position(H, W)

    return Object(Sprite(shape, color), pos)
end

make_objects = Map(make_object)

@gen function init_model(H,W,var)

    N ~ poisson(5)
    objs = {:init_objs} ~  make_objects(collect(1:N), [H for _ in 1:N], [W for _ in 1:N])

    rendered = draw(H, W, objs)
    {:observed_image} ~ image_likelihood(rendered, var)

    State(objs)
end

@gen (static) function model(H, W, T)

    var = .1
    init_state = {:init} ~ init_model(H,W,var)
    state = {:steps} ~ unfold_step(T-1, init_state, H, W, var)

    return state
end






function sim(T)
    (trace, _) = generate(model, (100, 100, T))
    return trace
end


# @gen function foo(t::Int, prev_state::State)

# end


# grid([Gen.simulate(model, (66,141,50)) for _=1:4])


# trace,_ = Gen.generate(model, (210, 160, 100));

# trace = Gen.simulate(model, (210, 160, 50));

# gif_of_trace(trace)

# grid([Gen.simulate(model, (66,141,50)) for _=1:4], annotate=true)

end # module Model