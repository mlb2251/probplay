
using Zygote
using Functors

include("html.jl")

struct Vec
    y::Float64
    x::Float64
end

mutable struct Gauss2D
    pos :: Vec
    scale_y::Float64
    scale_x::Float64
    angle::Float64
    opacity::Float64
    r :: Float64
    g :: Float64
    b :: Float64
end


function test3()
    fresh()
    H,W = 15,15

    targets = [rand_gaussian(H,W) for _ in 1:10]

    canvas = zeros(Float64, 3, H, W)
    target = draw_region(canvas, targets, 1, H, 1, W)
    html_body(html_img(canvas, width="400px"))

    gaussians = [rand_gaussian(H,W) for _ in 1:20]
    # @show Functors.functor(z)

    lr = .001

    for i in 1:50
        @show i
        canvas = zeros(Float64, 3, H, W)
        draw_region(canvas, gaussians, 1, H, 1, W)
        html_body(html_img(canvas, width="400px"))
    

        g = gradient(gaussians) do gaussians
            # draw_region(canvas, [z], 1, H, 1, W)
            loss = 0.
            for py in 1:H
                for px in 1:W
                    r,g,b = draw_pixel(gaussians, px, py)
                    # loss += (r - 1)^2 + (g - py/H)^2 + (b - 1)^2
                    loss += (target[1,py,px] - r)^2 + (target[2,py,px] - g)^2 + (target[3,py,px] - b)^2
                end
            end
            # z.pos.x #+ 2* z.scale_x
            loss
        end
        # @show length(g[1])
        g = fmap(x -> if isnothing(x) 0. else x end, g[1])

        for (i,z) in enumerate(gaussians)
            grads = g[i]
            z.pos = Vec(z.pos.y - lr * grads[:pos][:y], z.pos.x - lr * grads[:pos][:x])
            z.scale_y -= lr * grads[:scale_y]
            z.scale_x -= lr * grads[:scale_x]
            z.angle -= lr * grads[:angle]
            z.opacity -= lr * grads[:opacity]
            z.opacity = clamp(z.opacity, .1, 1.)
            z.r -= lr * grads[:r]
            z.g -= lr * grads[:g]
            z.b -= lr * grads[:b]
        end

    end

    render()


    # fmap(Functors.functor(z), g[1]) do z, g
        # z.pos.x -= lr * g
        # println(z, " -> ", g)
    # end
end

function rand_gaussian(H,W)
    Gauss2D(
        Vec(rand()*H, rand()*W),
        rand()*5,
        rand()*5,
        rand()*2pi,
        rand(),
        rand(),
        rand(),
        rand()
    )
end

function test()
    fresh()
    H,W = 100,100

    for _ in 1:4

    gaussians = [
        # Gauss2D(Vec(60.5,50), 10, 20, 1., 1, 1, 1, 1),
        # Gauss2D(Vec(60.5,50), 10, 20, 1., 1, 1, 1, 1),
        # Gauss2D(Vec(80.5,30), 5, 10, 1., 1, 1, 1, 1),
        # Gauss2D(Vec(140.5,30), 3, 3, 1., 1, 1, 1, 1),
        # Gauss2D(Vec(170.5,30), 30, 3, 0., 1, 1, 1, 1),
        # Gauss2D(Vec(140.5,60), 3, 3, 1., .5, 1, 1, 1),
        # Gauss2D(Vec(140.5,60), 1000, 1000, 1., 1., .7, 0., 0.4)
        rand_gaussian(H,W) for _ in 1:100
    ]

        canvas = zeros(Float64, 3, H*10, W*10)
        draw_region(canvas, gaussians, 1, H, 1, W)
        html_body(html_img(canvas, width="400px"))    
    end



    # for scale in 1:5
    #     canvas = zeros(Float64, 3, div(H,scale), div(W,scale))
    #     draw_region(canvas, gaussians, 1, H, 1, W)
    #     html_body(html_img(canvas, width="400px"))    
    # end

    # for scale in 1:5
    #     canvas = zeros(Float64, 3, H*scale, W*scale)
    #     draw_region(canvas, gaussians, 1, H, 1, W)
    #     html_body(html_img(canvas, width="400px"))    
    # end

    render()
end

function test2()
    fresh()
    H,W = 100,100
    canvas = zeros(Float64, 3, H, W)
    white_canvas = ones(Float64, 3, H, W)
    gaussians = [
        rand_gaussian(H,W) for _ in 1:60
    ]
    draw_region(canvas, gaussians, 1, H, 1, W)
    html_body(html_img(canvas, width="400px"))



    withgradient(params) do 
        
    end


    render()
end


function draw_region(canvas, gaussians, ymin, ymax, xmin, xmax)
    C,H,W = size(canvas)

    for y in 1:H
        for x in 1:W
            py = (y-1)/(H-1) * (ymax-ymin) + ymin
            px = (x-1)/(W-1) * (xmax-xmin) + xmin
            r,g,b = draw_pixel(gaussians, px, py)
            canvas[:,y,x] += [r,g,b]
        end
    end

    canvas 
end

function draw_pixel(gaussians, px, py)
    density_per_unit_area = 50.
    T = 1. # transmittance
    r = 0.
    g = 0.
    b = 0.
    for gauss in gaussians
        # get position relative to gaussian
        x2 = px-gauss.pos.x
        y2 = py-gauss.pos.y
        # now rotate
        x = x2*cos(gauss.angle) - y2*sin(gauss.angle)
        y = x2*sin(gauss.angle) + y2*cos(gauss.angle)

        # density_x = exp( - x^2 / (g.scale_x^2 * 2) ) / (sqrt(2pi) * g.scale_x)
        # density_y = exp( - y^2 / (g.scale_y^2 * 2) ) / (sqrt(2pi) * g.scale_y)
        # density = density_x * density_y
        # density = exp(-(x^2/g.scale_x^2 + y^2/g.scale_y^2)/2) / (2pi * g.scale_x * g.scale_y)

        # unnormalized density, so x=0 y=0 yields 1 meaning gaussians dont get spread out as they expand. Equivalent to scaled gaussian.
        density = exp(-(x^2/gauss.scale_x^2 + y^2/gauss.scale_y^2) / 2) / (2pi * gauss.scale_x * gauss.scale_y) * density_per_unit_area

        # density = exp(-(x^2/g.scale_x + y^2/g.scale_y))
        alpha = gauss.opacity * density
        alpha = clamp(alpha, 0., .999)
        r += T * alpha * gauss.r
        g += T * alpha * gauss.g
        b += T * alpha * gauss.b
        T *= 1. - alpha
        T < 0.01 && break
    end
    r = clamp(r, 0., 1.)
    g = clamp(g, 0., 1.)
    b = clamp(b, 0., 1.)
    return (r,g,b)
end
