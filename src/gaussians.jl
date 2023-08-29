

module Gaussians

using Zygote
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


function bench_loop()
    # device!(1)
    GC.gc(true)

    # CUDA.memory_status()
    # CUDA.memory_status()
    H,W = 100,100
    ITERS = 400
    N_GAUSS = 100
    total = 0.

    GC.gc(true)

    # sketch5: 5.042960s at 100x100 with 100 gauss and 400 iters
    # M1: 2.43s
    println("CPU")
    function inner_cpu()
        canvas = Array{Float32}(undef, (3, H, W))
        gaussians = [
            rand_igaussian2(H,W) for _ in 1:N_GAUSS
        ]
        draw_region(canvas, gaussians, 1, H, 1, W);
        maximum(canvas)
    end
    inner_cpu()
    @time for i in 1:ITERS
        total += inner_cpu()
    end

    GC.gc(true)


    converter = if CUDA.functional(); CuArray else MtlArray end
    # canvas_gpu = converter(canvas)
    # gaussians_gpu = converter(gaussians)


    # sketch5: 0.064943s at 100x100 with 100 gauss and 400 iters
    # M1: 0.41s
    println("GPU")
    function inner_gpu()
        canvas_gpu = converter{Float32}(undef, (3, H, W))
        gaussians = [rand_igaussian2(H,W) for _ in 1:N_GAUSS]
        gaussians_gpu = converter(gaussians)
        draw_region(canvas_gpu, gaussians_gpu, 1, H, 1, W);
        canvas = Array(canvas_gpu)
        maximum(canvas)
    end
    inner_gpu()
    @time for i in 1:ITERS
        total += inner_gpu()
    end




    GC.gc(true)
    # CUDA.memory_status()

    return
end


function bench_all()
    H,W = 200,200
    canvas = Array{Float32}(undef, 3, H, W)
    CUDA.functional() && CUDA.device!(1)
    GC.gc(true)

    gaussians = [rand_igaussian2(H,W) for _ in 1:100]

    # sketch5: 35.302ms @ 200x200 with 100 gaussians
    # M1: 20.711ms
    println("\ncpu")
    t=@benchmark draw_region($(canvas), $(gaussians), 1, $(H), 1, $(W))
    show(stdout, "text/plain", t)

    GC.gc(true)
    CUDA.functional() && CUDA.memory_status()

    # sketch5: 46.994 μs μs @ 200x200
    # M1: 576.362 μs @ 200x200
    converter = if CUDA.functional(); CuArray else MtlArray end
    canvas_gpu = converter(canvas)
    gaussians_gpu = converter(gaussians)
    t=@benchmark draw_region($(canvas_gpu), $(gaussians_gpu), 1, $(H), 1, $(W));
    println("\ngpu kernel")
    show(stdout, "text/plain", t)
    
    # fresh()
    # html_body(html_img(Array(canvas), width="400px"))
    # html_body(html_img(Array(canvas_cu), width="400px"))
    # render(;publish=true)

    GC.gc(true)
    CUDA.functional() && CUDA.memory_status()

    nothing
end


function test_all()
    fresh()
    H,W = 100,100
    canvas = Array{Float32}(undef, 3, H, W)
    gaussians = [
        rand_igaussian2(H,W) for _ in 1:100
    ]

    draw_region(canvas, gaussians, 1, H, 1, W);
    html_body(html_img(Float64.(canvas), width="400px"))

    canvas = Array{Float32}(undef, 3, H, W)

    converter = if CUDA.functional(); CuArray else MtlArray end

    canvas_gpu = converter(canvas)
    gaussians_gpu = converter(gaussians)
    draw_region(canvas_gpu, gaussians_gpu, 1, H, 1, W);

    canvas = Array(canvas_gpu)

    html_body(html_img(Float64.(canvas), width="400px"))
    render(publish=true)

    nothing
end

function test_ad()
    fresh()
    H,W = 20,20

    target = Array{Float32}(undef, 3, H, W)
    target_gaussians = [rand_igaussian2(H,W) for _ in 1:100]
    draw_region(target, target_gaussians, 1, H, 1, W);


    canvas = Array{Float32}(undef, 3, H, W)
    gaussians = [rand_igaussian2(H,W) for _ in 1:100]

    g = Zygote.gradient(gaussians) do gaussians
        draw_region(canvas, gaussians, 1, H, 1, W);
        sum(canvas .- target)
    end



    # html_body(html_img(Float64.(canvas), width="400px"))
    # render(publish=true)

    nothing

end



function draw_region(canvas, gaussians, ymin, ymax, xmin, xmax)
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))
    
    launch_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, threads, blocks)
end

function launch_kernel(canvas::MtlArray, gaussians, ymin, ymax, xmin, xmax, threads, blocks)
    Metal.@sync begin
        @metal threads=threads groups=blocks draw_pixel_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, length(gaussians))
    end
end

function launch_kernel(canvas::CuArray, gaussians, ymin, ymax, xmin, xmax, threads, blocks)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks draw_pixel_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, length(gaussians))
    end
end

function launch_kernel(canvas::Array, gaussians, ymin, ymax, xmin, xmax, threads, blocks)
    draw_pixel_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, length(gaussians))
end



function draw_pixel_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, N; target=nothing)

    _,H,W = size(canvas)

    ix,iy,stride_x,stride_y = get_pos_stride(canvas)
    density_per_unit_area = 50.0f0

    pixel_height = Float32(ymax-ymin)/H
    pixel_width = Float32(xmax-xmin)/W

    for cy in iy:stride_y:H
        for cx in ix:stride_x:W
            # calculate where to cast our ray for the pixel at (cy,cx) in the canvas assuming
            # (1,1) on the canvas in (ymin,xmin) in object space and (H,W) is (xmax,ymax)
            py = ymin + pixel_height * (cy-1)
            px = xmin + pixel_width * (cx-1)

            r,g,b = render_pixel(py,px,gaussians, N, density_per_unit_area)
            @inbounds canvas[1,cy,cx] = r
            @inbounds canvas[2,cy,cx] = g
            @inbounds canvas[3,cy,cx] = b
        end
    end

    return nothing
end

@inline function render_pixel(py,px,gaussians, N, density_per_unit_area)
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

        alpha = gauss.opacity * Float32(density)
        alpha = clamp(alpha, 0f0, .999f0)
        
        r += T * alpha * gauss.r
        g += T * alpha * gauss.g
        b += T * alpha * gauss.b
        T *= 1.0f0 - alpha
        T < 0.01f0 && break # doesnt really impact perf in gpu case
    end
    return clamp(r, 0f0, 1.0f0), clamp(g, 0f0, 1.0f0), clamp(b, 0f0, 1.0f0)
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




# function mae_kernel(xs, ys, var)
#     ix,iy,stride_x,stride_y = get_pos_stride(target)
#     for i = index:stride:length(target)
#         @inbounds target[i] = - (abs2((xs[i] - ys[i]) / var) + log(2π)) / 2 - log(var)
#     end
#     return nothing
# end




end