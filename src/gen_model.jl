
using Gen
using LinearAlgebra
using Images
using Distributions
using Plots





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

# bernoulli_2d(0.5, 10,20)


struct ImageLikelihood <: Gen.Distribution{Array} end

function Gen.logpdf(::ImageLikelihood, observed_image, rendered_image, var)
    # todo is this right?
    diff = observed_image - rendered_image
    # Gen.logpdf(Gen.MultivariateNormal, diff, zeros(size(diff)), var * Matrix(I, size(diff)))
    # sum bc independent events
    sum(Distributions.logpdf.(Normal(0, var), diff))
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




struct Sprite
    mask::Matrix{Bool}
    color::RGB{Float64}
end

struct Object
    sprite::Sprite
    x::Int
    y::Int
end


function canvas(height=210, width=160, background=RGB{Float64}(0,0,0))
    fill(background, height, width)
end

function draw!(canvas, obj::Object)
    sprite = obj.sprite
    for I in CartesianIndices(sprite.mask)
        i, j = Tuple(I)
        if sprite.mask[i,j]
            offy = obj.y+i-1
            offx = obj.x+j-1
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


@gen function model2(canvas_height, canvas_width, T)

    var = .1

    N ~ poisson(10)
    objs = Object[]

    # initialize objects
    for i in 1:N
        w = {(i => :width)} ~ uniform_discrete(1,100)
        h = {(i => :height)} ~ uniform_discrete(1,100)
        shape = {(i => :shape)} ~ bernoulli_2d(0.5, h,w)
        color = {(i => :color)} ~ rgb_dist()
        sprite = Sprite(shape, color)

        pos_y = {(i => :pos_y)} ~ uniform_discrete(1,canvas_height)
        pos_x = {(i => :pox_x)} ~ uniform_discrete(1,canvas_width)
        obj = Object(sprite, pos_x, pos_y)
        push!(objs, obj)
    end

    # render
    rendered = Array(channelview(draw!(canvas(canvas_height, canvas_width), objs)))
    observed_image ~ image_likelihood(rendered, var)

    for t in 1:T
        for i in 1:N
            dx = {i => t => :dx} ~ uniform_discrete(-3,3)
            dy = {i => t => :dy} ~ uniform_discrete(-3,3)
            objs[i] = Object(objs[i].sprite, objs[i].x + dx, objs[i].y + dy)
        end
        rendered = Array(channelview(draw!(canvas(canvas_height, canvas_width), objs)))
        observed_image = {t => :observed_image} ~ image_likelihood(rendered, var)
    end

    return
end


# trace,_ = Gen.generate(model2, (210, 160, 100));

function gif_of_trace(trace)
    T = get_args(trace)[3]

    @gif for t in 1:T
        observed = colorview(RGB,trace[t => :observed_image])

        # draw!(c, objs)
        plot(observed, xlims=(0,160), ylims=(0,210), ticks=true)
        annotate!(-60, 20, "Step: $t")
        annotate!(-60, 40, "Objects: $(trace[:N])")
    end
end

# gif_of_trace(trace)

function games()
    [x for x in readdir("out/gameplay") if occursin("v5",x)]
end

function frames(game)
    frames = length(["out/gameplay/$game/$x" for x in readdir("out/gameplay/$game") if occursin("png",x)])
    ["out/gameplay/$game/$i.png" for i in 1:frames]
end


# for frame in frames("ALE-Breakout-v5")
#    
# end



function plot_gameplay(game)
    imgs =  channelview.(load.(frames(game)))
    imgs = [Float64.(x) for x in imgs]

    @gif for t in 2:100
        img = colorview(RGB,imgs[t])
        p1 = plot(img, xlims=(0,160), ylims=(0,210), ticks=true)
        p2 = plot(img, xlims=(0,160), ylims=(0,210), ticks=true)
        plot(p1,p2)
    #     observed = colorview(RGB,trace[t => :observed_image])

    #     # draw!(c, objs)
    #     plot(observed, xlims=(0,160), ylims=(0,210), ticks=true)
        annotate!(-60, 20, "Step: $t")
    #     annotate!(-60, 40, "Objects: $(trace[:N])")
    end
end
