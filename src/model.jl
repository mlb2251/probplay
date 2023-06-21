
using Gen
using LinearAlgebra
using Images
using Distributions
using Plots
using AutoHashEquals
using HypertextLiteral
using Dates

include("html.jl")

@auto_hash_equals struct Position
    y::Int
    x::Int
end

struct Sprite
    mask::Matrix{Bool}
    color::RGB{Float64}
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

function Gen.logpdf(::ImageLikelihood, observed_image, rendered_image, var)
    diff = observed_image - rendered_image
    # Gen.logpdf(Gen.MultivariateNormal, diff, zeros(size(diff)), var * Matrix(I, size(diff)))
    # sum bc independent events
    # sum(Gen.logpdf.(Ref(normal),diff, 0, var))
    # sum(Distributions.logpdf.(Normal(0, var), diff))
    Gen.logpdf(broadcasted_normal, diff, zeros(Float64,size(diff)), var)
end

function Gen.random(::ImageLikelihood, rendered_image, var)
    noise = rand(Normal(0, var), size(rendered_image))
    # noise = mvnormal(zeros(size(rendered_image)), var * Maxxtrix(I, size(rendered_image)))
    rendered_image .+ noise
end

const image_likelihood = ImageLikelihood()
(::ImageLikelihood)(rendered_image, var) = random(ImageLikelihood(), rendered_image, var)

struct RGBDist <: Gen.Distribution{RGB} end

function Gen.logpdf(::RGBDist, rgb)
    0. # uniform distribution over unit cube has density 1
end

function Gen.random(::RGBDist)
    rand(RGB)
end

const rgb_dist = RGBDist()

(::RGBDist)() = random(RGBDist())


function canvas(height=210, width=160, background=RGB{Float64}(0,0,0))
    fill(background, height, width)
end

"""
renders an object on a canvas
"""
function draw!(canvas, obj::Object)
    sprite = obj.sprite
    for I in CartesianIndices(sprite.mask)
        i, j = Tuple(I)
        if sprite.mask[i,j]
            offy = obj.pos.y+i-1
            offx = obj.pos.x+j-1
            if offy > 0 && offy <= size(canvas,1) && offx > 0 && offx <= size(canvas,2)
                canvas[offy,offx] = sprite.color
            end
        end
    end
    canvas
end

function draw!(canvas, objs::Vector{Object})
    for obj in objs
        draw!(canvas, obj)
    end
    canvas
end

"""
The generative model
"""
@gen function model(canvas_height, canvas_width, T)

    var = .1

    N ~ poisson(5)
    objs = Object[]

    # initialize objects
    for i in 1:N
        w = {(i => :width)} ~ uniform_discrete(1,canvas_width)
        h = {(i => :height)} ~ uniform_discrete(1,canvas_height)
        shape = {(i => :shape)} ~ bernoulli_2d(0.5, h,w)
        color = {(i => :color)} ~ rgb_dist()
        sprite = Sprite(shape, color)

        pos = {(1 => i => :pos)} ~ uniform_position(canvas_height, canvas_width)

        obj = Object(sprite, pos)
        push!(objs, obj)
    end

    # render
    rendered = Array(channelview(draw!(canvas(canvas_height, canvas_width), objs)))
    observed_image = {1 => :observed_image} ~ image_likelihood(rendered, var)

    for t in 2:T
        for i in 1:N
            pos = {t => i => :pos} ~ uniform_drift_position(objs[i].pos, 3)
            objs[i] = Object(objs[i].sprite, pos)
        end
        rendered = Array(channelview(draw!(canvas(canvas_height, canvas_width), objs)))
        observed_image = {t => :observed_image} ~ image_likelihood(rendered, var)
    end

    return
end


# grid([Gen.simulate(model, (66,141,50)) for _=1:4])


# trace,_ = Gen.generate(model, (210, 160, 100));

# trace = Gen.simulate(model, (210, 160, 50));

# gif_of_trace(trace)

# grid([Gen.simulate(model, (66,141,50)) for _=1:4], annotate=true)