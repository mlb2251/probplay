

module Gaussians

# using Zygote
# using Functors
using CUDA
using Metal
# using ReverseDiff
using BenchmarkTools

include("html.jl")

struct Vec
    y::Float32
    x::Float32
end

mutable struct Gauss2D
    pos :: Vec
    scale_y::Float32
    scale_x::Float32
    angle::Float32
    opacity::Float32
    r :: Float32
    g :: Float32
    b :: Float32
end

struct IGauss2D
    pos :: Vec
    scale_y::Float32
    scale_x::Float32
    angle::Float32
    opacity::Float32
    r :: Float32
    g :: Float32
    b :: Float32
end

struct IGauss2D2
    pos :: Vec
    scale_y::Float32
    scale_x::Float32
    cos_angle::Float32
    sin_angle::Float32
    opacity::Float32
    r :: Float32
    g :: Float32
    b :: Float32
end


function test_zygote()
    fresh()
    H,W = 15,15

    targets = [rand_gaussian(H,W) for _ in 1:10]

    canvas = zeros(Float32, 3, H, W)
    target = draw_region(canvas, targets, 1, H, 1, W)
    html_body(html_img(canvas, width="400px"))

    gaussians = [rand_gaussian(H,W) for _ in 1:20]
    # @show Functors.functor(z)

    lr = .001

    for i in 1:50
        @show i
        canvas = zeros(Float32, 3, H, W)
        draw_region(canvas, gaussians, 1, H, 1, W)
        html_body(html_img(canvas, width="400px"))
    

        g = Zygote.gradient(gaussians) do gaussians
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

function rand_igaussian(H,W)
    IGauss2D(
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

function rand_igaussian2(H,W)
    xrot = rand()*2 - 1
    yrot = rand()*2 - 1
    norm = sqrt(xrot^2 + yrot^2)
    xrot /= norm
    yrot /= norm
    @assert isapprox(1, xrot^2 + yrot^2)
    IGauss2D2(
        Vec(rand()*H, rand()*W),
        rand()*5,
        rand()*5,
        xrot,
        yrot,
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

        canvas = zeros(Float32, 3, H*10, W*10)
        draw_region(canvas, gaussians, 1, H, 1, W)
        html_body(html_img(canvas, width="400px"))    
    end



    # for scale in 1:5
    #     canvas = zeros(Float32, 3, div(H,scale), div(W,scale))
    #     draw_region(canvas, gaussians, 1, H, 1, W)
    #     html_body(html_img(canvas, width="400px"))    
    # end

    # for scale in 1:5
    #     canvas = zeros(Float32, 3, H*scale, W*scale)
    #     draw_region(canvas, gaussians, 1, H, 1, W)
    #     html_body(html_img(canvas, width="400px"))    
    # end

    render()
end

function test2()
    fresh()
    H,W = 100,100
    canvas = zeros(Float32, 3, H, W)
    white_canvas = ones(Float32, 3, H, W)
    gaussians = [
        rand_gaussian(H,W) for _ in 1:60
    ]
    draw_region(canvas, gaussians, 1, H, 1, W)
    html_body(html_img(canvas, width="400px"))



    withgradient(params) do 
        
    end


    render()
end

function test_cuda()
    H,W = 100,100
    canvas = zeros(Float32, 3, H, W)
    device!(3)
    canvas_cu = CuArray(canvas)
    gaussians = [
        rand_igaussian2(H,W) for _ in 1:100
    ]

    # CUDA.@sync begin
    #     canvas_cu = CuArray{Float32}(undef, (3, H, W))
    #     gaussians_cu = CuArray(gaussians)
    #     draw_region_kernel(canvas_cu, gaussians_cu, 1, H, 1, W);
    #     canvas = Array(canvas_cu)
    # end

    # ~ 1.3ms for 100x100 with 100 gauss
    @time CUDA.@sync begin
        canvas_cu = CuArray{Float32}(undef, (3, H, W))
        gaussians_cu = CuArray(gaussians)
        draw_region_kernel(canvas_cu, gaussians_cu, 1, H, 1, W);
        canvas = Array(canvas_cu)
    end


    # @time draw_region(canvas_cu, gaussians, 1, H, 1, W)
    # @time 
    # fresh()
    # html_body(html_img(Array(canvas_cu), width="400px"))
    # render(;publish=true)

    nothing
end


function loop_bench_cuda()
    # device!(1)
    GC.gc(true)

    # CUDA.memory_status()
    # CUDA.memory_status()
    H,W = 100,100
    ITERS = 400
    N_GAUSS = 100
    total = 0.

    # sketch5: 0.280410s at 100x100 with 100 gauss and 400 iters
    # M1: 0.41s
    println("GPU")
    @time for i in 1:ITERS
        # canvas = Array{Float32}(undef, (3, H, W))
        # canvas_cu = CuArray(canvas)
        # CUDA.@sync begin
        #     canvas_cu = CuArray{Float32}(undef, (3, H, W))
        #     gaussians = [rand_igaussian2(H,W) for _ in 1:N_GAUSS]
        #     gaussians_cu = CuArray(gaussians)
        #     draw_region_kernel(canvas_cu, gaussians_cu, 1, H, 1, W);
        #     canvas = Array(canvas_cu)
        #     total += maximum(canvas)
        # end

        Metal.@sync begin
            canvas_mtl = MtlArray{Float32}(undef, (3, H, W))
            gaussians = [rand_igaussian2(H,W) for _ in 1:N_GAUSS]
            gaussians_mtl = MtlArray(gaussians)
            draw_region_kernel_mtl(canvas_mtl, gaussians_mtl, 1, H, 1, W);
            canvas = Array(canvas_mtl)
            total += maximum(canvas)
        end



    end



    # sketch5: 6.783885s at 100x100 with 100 gauss and 400 iters
    # M1: 4.0s
    println("CPU")
    @time for i in 1:ITERS
        canvas = Array{Float32}(undef, (3, H, W))
        gaussians = [
            rand_igaussian2(H,W) for _ in 1:N_GAUSS
        ]
        draw_region(canvas, gaussians, 1, H, 1, W);
        total += maximum(canvas)
    end



    GC.gc(true)
    # CUDA.memory_status()

    return
end


function bench_cuda()
    H,W = 200,200
    canvas = zeros(Float32, 3, H, W)
    # device!(1)
    GC.gc(true)
    # CUDA.memory_status()

    gaussians = [rand_igaussian2(H,W) for _ in 1:100]

    # sketch5: 64.286ms @ 200x200 with 100 gaussians
    # M1: 38.422 ms
    println("\ncpu")
    t=@benchmark draw_region($(canvas), $(gaussians), 1, $(H), 1, $(W))
    show(stdout, "text/plain", t)

    # 926.294 ms @ 200x200 (becomes like 8 seconds if you use gaussians_cu)
    # println("\ngpu")
    # t=@benchmark draw_region($(canvas_cu), $(gaussians), 1, $(H), 1, $(W))
    # show(stdout, "text/plain", t)

    GC.gc(true)
    # CUDA.memory_status()

    # sketch5: 930.345 μs @ 200x200
    # println("\ngpu kernel")
    # @time canvas_cu = CuArray(canvas)
    # gaussians_cu = CuArray(gaussians)
    # t=@benchmark draw_region_kernel($(canvas_cu), $(gaussians_cu), 1, $(H), 1, $(W));
    # show(stdout, "text/plain", t)

    canvas_mtl = MtlArray(canvas)
    gaussians_mtl = MtlArray(gaussians)

    # M1: 591.558 μs @ 200x200
    println("\ngpu kernel")
    t=@benchmark draw_region_kernel_mtl($(canvas_mtl), $(gaussians_mtl), 1, $(H), 1, $(W));
    show(stdout, "text/plain", t)

    # draw_region_kernel(canvas_cu, gaussians_cu, 1, H, 1, W);

    # draw_region_kernel(canvas_cu, gaussians_cu, 1, H, 1, W);

    
    # fresh()
    # html_body(html_img(Array(canvas), width="400px"))
    # html_body(html_img(Array(canvas_cu), width="400px"))
    # render(;publish=true)

    GC.gc(true)
    # CUDA.memory_status()

    nothing
end


function test_metal()
    H,W = 100,100
    canvas = zeros(Float32, 3, H, W)
    gaussians = [
        rand_igaussian2(H,W) for _ in 1:100
    ]

    # CUDA.@sync begin
    #     canvas_cu = CuArray{Float32}(undef, (3, H, W))
    #     gaussians_cu = CuArray(gaussians)
    #     draw_region_kernel(canvas_cu, gaussians_cu, 1, H, 1, W);
    #     canvas = Array(canvas_cu)
    # end

    # ~ 1.3ms for 100x100 with 100 gauss
    @time Metal.@sync begin
        canvas_mtl = MtlArray{Float32}(undef, (3, H, W))
        gaussians_mtl = MtlArray(gaussians)
        draw_region_kernel_mtl(canvas_mtl, gaussians_mtl, 1, H, 1, W);
        canvas = Array(canvas_mtl)
    end


    # @time draw_region(canvas_cu, gaussians, 1, H, 1, W)
    # @time 
    fresh()
    html_body(html_img(Float64.(canvas), width="400px"))
    render(;#=publish=true=#)

    nothing
end


function draw_region_kernel_mtl(canvas, gaussians, ymin, ymax, xmin, xmax)
    C,H,W = size(canvas)
    threads_x = 16
    blocks_x = cld(W, threads_x)
    threads_y = 16
    blocks_y = cld(H, threads_y)
    Metal.@sync begin
        @metal threads=(threads_x,threads_y) groups=(blocks_x,blocks_y) draw_pixel_kernel_mtl(canvas, gaussians, ymin, ymax, xmin, xmax, length(gaussians))
    end
end

function draw_region_kernel(canvas, gaussians, ymin, ymax, xmin, xmax)
    C,H,W = size(canvas)
    threads_x = 16
    blocks_x = cld(W, threads_x)
    threads_y = 16
    blocks_y = cld(H, threads_y)
    CUDA.@sync begin
        @cuda threads=(threads_x,threads_y) blocks=(blocks_x,blocks_y) draw_pixel_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, length(gaussians))
    end
end

function draw_pixel_kernel_mtl(canvas, gaussians, ymin, ymax, xmin, xmax, N)

    _,H,W = size(canvas)

    ix,iy,stride_x,stride_y = get_pos_stride(canvas)

    for cy in iy:stride_y:H
        for cx in ix:stride_x:W
            py = (Float32(cy)-1)/(H-1) * (ymax-ymin) + ymin
            px = (Float32(cx)-1)/(W-1) * (xmax-xmin) + xmin

            density_per_unit_area = 50.0f0
            T = 1.0f0 # transmittance
            r = 0.0f0
            g = 0.0f0
            b = 0.0f0
            for G in 1:N
                gauss = gaussians[G]
                # get position relative to gaussian
                x2 = px-gauss.pos.x
                y2 = py-gauss.pos.y

                # now rotate
                x = x2*gauss.cos_angle - y2*gauss.sin_angle
                y = x2*gauss.sin_angle + y2*gauss.cos_angle
                
                density = exp(-((x/gauss.scale_x)^2 + (y/gauss.scale_y)^2) / 2) / (2 * 3.141592653589f0 * gauss.scale_x * gauss.scale_y) * density_per_unit_area
                # density = exp(((x)))
        
                alpha = gauss.opacity * Float32(density)
                alpha = clamp(alpha, 0f0, .999f0)
                
                r += T * alpha * gauss.r
                g += T * alpha * gauss.g
                b += T * alpha * gauss.b
                T *= 1.0f0 - alpha
                T < 0.01f0 && break # doesnt really impact perf in gpu case
            end
            @inbounds canvas[1,cy,cx] = clamp(r, 0f0, 1.0f0)
            @inbounds canvas[2,cy,cx] = clamp(g, 0f0, 1.0f0)
            @inbounds canvas[3,cy,cx] = clamp(b, 0f0, 1.0f0)
        end
    end

    return nothing
end

@inline function get_pos_stride(::CuDeviceArray)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    stride_x = gridDim().x * blockDim().x
    stride_y = gridDim().y * blockDim().y

    return ix,iy,stride_x,stride_y
end

@inline function get_pos_stride(::MtlDeviceArray)
    ix,iy = Metal.thread_position_in_grid_2d()
    stride_x = threads_per_threadgroup_2d().x * threadgroups_per_grid_2d().x
    stride_y = threads_per_threadgroup_2d().y * threadgroups_per_grid_2d().y

    return ix,iy,stride_x,stride_y
end

@inline function get_pos_stride(::Array)
    return 1,1,1,1
end


function draw_pixel_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, N)

    _,H,W = size(canvas)

    ix,iy,stride_x,stride_y = get_pos_stride(canvas)

    for cy in iy:stride_y:H
        for cx in ix:stride_x:W
            py = (cy-1)/(H-1) * (ymax-ymin) + ymin
            px = (cx-1)/(W-1) * (xmax-xmin) + xmin

            density_per_unit_area = 50.
            T = 1. # transmittance
            r = 0.
            g = 0.
            b = 0.
            for G in 1:N
                gauss = gaussians[G]
                # get position relative to gaussian
                x2 = px-gauss.pos.x
                y2 = py-gauss.pos.y

                # now rotate
                x = x2*gauss.cos_angle - y2*gauss.sin_angle
                y = x2*gauss.sin_angle + y2*gauss.cos_angle
                
                density = exp(-((x/gauss.scale_x)^2 + (y/gauss.scale_y)^2) / 2) / (2pi * gauss.scale_x * gauss.scale_y) * density_per_unit_area
        
                alpha = gauss.opacity * density
                alpha = clamp(alpha, 0., .999)
                
                r += T * alpha * gauss.r
                g += T * alpha * gauss.g
                b += T * alpha * gauss.b
                T *= 1. - alpha
                T < 0.01 && break # doesnt really impact perf in gpu case
            end
            @inbounds canvas[1,cy,cx] = clamp(r, 0., 1.)
            @inbounds canvas[2,cy,cx] = clamp(g, 0., 1.)
            @inbounds canvas[3,cy,cx] = clamp(b, 0., 1.)
        end
    end

    return nothing
end




function draw_region(canvas, gaussians, ymin, ymax, xmin, xmax)

    draw_pixel_kernel_mtl(canvas, gaussians, ymin, ymax, xmin, xmax, length(gaussians))

    # C,H,W = size(canvas)

    # for y in 1:H
    #     for x in 1:W
    #         py = (y-1)/(H-1) * (ymax-ymin) + ymin
    #         px = (x-1)/(W-1) * (xmax-xmin) + xmin
    #         draw_pixel(canvas, gaussians, x, y, px, py)
    #         # r,g,b = draw_pixel(gaussians, px, py)
    #         # canvas[:,y,x] = draw_pixel(gaussians, px, py)
    #         # canvas[1,y,x] = r
    #         # canvas[2,y,x] = g
    #         # canvas[3,y,x] = b
    #     end
    # end

    # canvas 
end

function draw_pixel(canvas, gaussians, cx, cy, px, py)
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
        # x = x2*cos(gauss.angle) - y2*sin(gauss.angle)
        # y = x2*sin(gauss.angle) + y2*cos(gauss.angle)
        x = x2*gauss.cos_angle - y2*gauss.sin_angle
        y = x2*gauss.sin_angle + y2*gauss.cos_angle

        # density_x = exp( - x^2 / (g.scale_x^2 * 2) ) / (sqrt(2pi) * g.scale_x)
        # density_y = exp( - y^2 / (g.scale_y^2 * 2) ) / (sqrt(2pi) * g.scale_y)
        # density = density_x * density_y
        # density = exp(-(x^2/g.scale_x^2 + y^2/g.scale_y^2)/2) / (2pi * g.scale_x * g.scale_y)

        # unnormalized density, so x=0 y=0 yields 1 meaning gaussians dont get spread out as they expand. Equivalent to scaled gaussian.
        density = exp(-((x/gauss.scale_x)^2 + (y/gauss.scale_y)^2) / 2) / (2pi * gauss.scale_x * gauss.scale_y) * density_per_unit_area

        # density = exp(-(x^2/g.scale_x + y^2/g.scale_y))
        alpha = gauss.opacity * density
        alpha = clamp(alpha, 0., .999)
        
        r += T * alpha * gauss.r
        g += T * alpha * gauss.g
        b += T * alpha * gauss.b
        T *= 1. - alpha
        T < 0.01 && break
    end

    @inbounds canvas[1,cy,cx] = clamp(r, 0., 1.)
    @inbounds canvas[2,cy,cx] = clamp(g, 0., 1.)
    @inbounds canvas[3,cy,cx] = clamp(b, 0., 1.)
    return nothing
end










export test


# function test()
#     CUDA.versioninfo();
#     @show CUDA.device();
#     a = 1
# end



# 1.468 ms
function logpdf(observed_image, rendered_image, var)
    # precomputing log(var) and assuming mu=0 both give speedups here
    log_var = log(var)
    sum(i -> - (@inbounds abs2((observed_image[i] - rendered_image[i]) / var) + log(2π)) / 2 - log_var, eachindex(observed_image))
end

# 9.985 ms; cu = 385.682 μs
function logpdf2(observed_image, rendered_image, var)
    # precomputing log(var) and assuming mu=0 both give speedups here
    log_var = log(var)
    sum( @. - (abs2((observed_image - rendered_image) / var) + log(2π)) / 2 - log_var)
end

# 26.353 ms; cu = 521.248 μs
function logpdf3(observed_image, rendered_image, var)
    sum( @. - (abs2((observed_image - rendered_image) / var) + log(2π)) / 2 - log(var))
end

# 454.811 μs
function logpdf_unreduced(observed_image, rendered_image, var)
    @. - (abs2((observed_image - rendered_image) / var) + log(2π)) / 2 - log(var)
end


# 442.703 μs
function logpdf_unreduced_prealloc(target, observed_image, rendered_image, var)
    @. target .= - (abs2((observed_image - rendered_image) / var) + log(2π)) / 2 - log(var)
end

function kernel(target, xs, ys, var)
    # index = threadIdx().x
    # stride = blockDim().x
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(target)
        @inbounds target[i] = - (abs2((xs[i] - ys[i]) / var) + log(2π)) / 2 - log(var)
    end
    return nothing
end

# 364.323 μs without sum and 422.796 μs with sum
function bench2!(target, xs, ys, var)
    # compile kernel but don't launch it
    kern = @cuda launch=false kernel(target, xs, ys, var)
    config = launch_configuration(kern.fun)
    threads = min(length(target), config.threads)
    blocks = cld(length(target), threads)
    # numblocks = ceil(Int, length(target)/256)

    CUDA.@sync begin
        kern(target, xs, ys, var; threads, blocks)
        # @cuda threads=256 blocks=numblocks kernel(target, xs, ys, var)
    end
end


export logpdf_tests
function logpdf_tests()
    x = rand(3,1000,1000) # 3 million element array
    y = rand(3,1000,1000)
    var = 0.1

    println("logpdf()")
    t = @benchmark logpdf($(x),$(y),$(var))
    show(stdout, "text/plain", t)

    println("logpdf2()")
    t = @benchmark logpdf2($(x),$(y),$(var))
    show(stdout, "text/plain", t)

    println("logpdf2() cuda")
    t = @benchmark logpdf3($(CuArray(x)),$(CuArray(y)),$(var))
    show(stdout, "text/plain", t)    

    nothing
end





end