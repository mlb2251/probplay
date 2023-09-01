

module Gaussians

using CUDA
using Metal
using BenchmarkTools
using ForwardDiff

include("html.jl")

const G_Y = 1
const G_X = 2
const G_SCALE_Y = 3
const G_SCALE_X = 4
const G_COS_ANGLE = 5
const G_SIN_ANGLE = 6
const G_OPACITY = 7
const G_R = 8
const G_G = 9
const G_B = 10

const G_PARAMS = 10

# struct Gauss
#     pos :: Vec
#     scale_y::Float32
#     scale_x::Float32
#     cos_angle::Float32
#     sin_angle::Float32
#     opacity::Float32
#     r :: Float32
#     g :: Float32
#     b :: Float32
# end

zero_gauss() = zeros(Float32, G_PARAMS)
zero_gauss(N) = zeros(Float32, G_PARAMS, N)

function rand_gauss(H,W,N)
    stack([rand_gauss(H,W) for _ in 1:N])
end

function rand_gauss(H,W)
    cos_angle = rand()*2 - 1
    sin_angle = rand()*2 - 1
    norm = sqrt(cos_angle^2 + sin_angle^2)
    cos_angle /= norm
    sin_angle /= norm
    @assert isapprox(1, cos_angle^2 + sin_angle^2)
    Float32[
        rand()*H,
        rand()*W,
        rand()*5 * H/100,
        rand()*5 * W/100,
        cos_angle,
        sin_angle,
        rand(),
        rand(),
        rand(),
        rand()
    ]
end

function bench_loop()
    # device!(1)
    GC.gc(true)

    # CUDA.memory_status()
    # CUDA.memory_status()
    H,W = 100,100
    ITERS = 400
    N = 100
    total = 0.

    GC.gc(true)

    # sketch5: 5.042960s at 100x100 with 100 gauss and 400 iters
    # M1: 2.43s
    println("CPU")
    function inner_cpu()
        canvas = Array{Float32}(undef, (3, H, W))
        gaussians = rand_gauss(H,W,N)
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
        gaussians = [rand_igaussian2(H,W) for _ in 1:N]
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

    # sketch5: 46.994 μs @ 200x200 (ranges from 40-70 microseconds really)
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

dual_type(valtype,::ForwardDiff.GradientConfig{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T,valtype,N}

function test_gradients()

    # target = Array(Float32.(channelview(load("little_tony.png"))))[1:3,:,:]
    # html_body(html_img(Float64.(target), width="400px"))
    # _,H,W = size(target)



    H,W = 100,100
    N = 10
    canvas = zeros(Float32, 3, H, W)

    target = zeros(Float32, 3, H, W)
    target[1,:,:] .= 1.

    orig_gaussians = rand_gauss(H,W,N)
    gaussians = copy(orig_gaussians)

    chunk_sz = min(H*W, 8)
    cfg = ForwardDiff.GradientConfig(nothing, gaussians, ForwardDiff.Chunk{chunk_sz}())
    canvas = zeros(dual_type(Float32,cfg), 3, H, W)

    println("CPU gradients...")
    lr = .01f0
    grad_step(canvas, target, gaussians, lr, cfg)
    @time for i in 1:10
        @show i
        grad_step(canvas, target, gaussians, 0f0, cfg)
        html_body(html_img(Float64.(ForwardDiff.value.(canvas)), width="400px"))
    end

    println("GPU gradients...")
    CUDA.functional() && CUDA.device!(1)
    CUDA.functional() && CUDA.memory_status()
    converter = if CUDA.functional(); CuArray else MtlArray end
    gaussians = converter(orig_gaussians)
    target = converter(target)
    canvas = converter(canvas)
    cfg = ForwardDiff.GradientConfig(nothing, gaussians, ForwardDiff.Chunk{chunk_sz}())

    lr = .01f0
    grad_step(canvas, target, gaussians, 0f0, cfg)
    @time for i in 1:50
        @show i
        grad_step(canvas, target, gaussians, lr, cfg)
        html_body(html_img(Float64.(ForwardDiff.value.(Array(canvas))), width="400px"))
    end

    nothing
end

function grad_step(canvas, target, gaussians, lr, cfg)
    # @show size(target) size(canvas) typeof(target) typeof(canvas) typeof(canvas .- target) typeof(target .- canvas) sizeof(canvas)
    # sum(abs.(canvas .- target))
    _,H,W = size(canvas)
    dgaussians = ForwardDiff.gradient(gaussians, cfg) do gaussians
        draw_region(canvas, gaussians, 1, H, 1, W)
        sum(abs.(target .- canvas))
        # 1f0
    end
    gaussians .-= lr .* dgaussians

    normalize_gaussians(gaussians)

    # possibly this stuff should go within the render function idk
    # for G in axes(gaussians,2)
    #     norm = sqrt(gaussians[G_COS_ANGLE,G]^2 + gaussians[G_SIN_ANGLE,G]^2)
    #     gaussians[G_COS_ANGLE,G] /= norm
    #     gaussians[G_SIN_ANGLE,G] /= norm
        
    #     gaussians[G_OPACITY,G] = clamp(gaussians[G_OPACITY,G], 0f0, 1f0)
    # end
end

function normalize_gaussians(gaussians::T) where T <: CuArray
    threads = 64
    blocks = cld(size(gaussians,2), threads)
    CUDA.@sync @cuda threads=threads blocks=blocks normalize_gaussians_kernel(gaussians)
end

function normalize_gaussians(gaussians::T) where T <: MtlArray
    threads = 64
    blocks = cld(size(gaussians,2), threads)
    Metal.@sync @metal threads=threads groups=blocks normalize_gaussians_kernel(gaussians)
end

function normalize_gaussians_kernel(gaussians::T) where T <: CuDeviceArray
    G = threadIdx().x
    if G > size(gaussians,2)
        return
    end
    normalize_gaussians_inner(gaussians, G)
    return
end

function normalize_gaussians_kernel(gaussians::T) where T <: MtlDeviceArray
    G = Metal.thread_position_in_grid_1d()
    if G > size(gaussians,2)
        return
    end
    normalize_gaussians_inner(gaussians, G)
    return
end

function normalize_gaussians(gaussians::T) where T <: Array
    for G in axes(gaussians,2)
        normalize_gaussians_inner(gaussians,G)
    end
end

@inline function normalize_gaussians_inner(gaussians, G)
    # norm rotation
    norm = sqrt(gaussians[G_COS_ANGLE,G]^2 + gaussians[G_SIN_ANGLE,G]^2)
    gaussians[G_COS_ANGLE,G] /= norm
    gaussians[G_SIN_ANGLE,G] /= norm
    
    # clamp opacity
    gaussians[G_OPACITY,G] = clamp(gaussians[G_OPACITY,G], 0f0, 1f0)
end



function test_render()
    CUDA.functional() && CUDA.device!(0)
    CUDA.functional() && CUDA.memory_status()
    H,W = 100,100
    N = 100
    # canvas = Array{Float32}(undef, 3, H, W)
    canvas = zeros(Float32, 3, H, W)

    gaussians = rand_gauss(H,W,N)

    println("CPU rendering...")
    draw_region(canvas, gaussians, 1, H, 1, W)
    html_body(html_img(Float64.(canvas), width="400px"))

    canvas = zeros(Float32,3, H, W)

    converter = if CUDA.functional(); CuArray else MtlArray end
    canvas_gpu = converter(canvas)
    gaussians_gpu = converter(gaussians)

    println("GPU render")
    draw_region(canvas_gpu, gaussians_gpu, 1, H, 1, W)

    canvas = Array(canvas_gpu)

    html_body(html_img(Float64.(canvas), width="400px"))
    nothing
end

function draw_region(canvas::T, gaussians, ymin, ymax, xmin, xmax) where T <: MtlArray
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))

    Metal.@sync begin
        @metal threads=threads groups=blocks draw_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
    return nothing
end

function draw_region(canvas::T, gaussians, ymin, ymax, xmin, xmax) where T <: CuArray
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks draw_kernel(canvas, gaussians, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
    return nothing
end

function draw_region(canvas::T, gaussians, ymin, ymax, xmin, xmax) where T <: Array
    _,H,W = size(canvas)

    for cx in 1:W, cy in 1:H
        draw_kernel_inner(canvas, gaussians, cx, cy, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
end


function draw_kernel(canvas::T, gaussians, ymin, ymax, xmin, xmax, N) where T <: CuDeviceArray
    # get which pixel of the canvas should this thread should draw
    cx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    draw_kernel_inner(canvas, gaussians, cx, cy, ymin, ymax, xmin, xmax, N)
    return nothing
end

function draw_kernel(canvas::T, gaussians, ymin, ymax, xmin, xmax, N) where T <: MtlDeviceArray
    # get which pixel of the canvas should this thread should draw
    cx,cy = Metal.thread_position_in_grid_2d()

    draw_kernel_inner(canvas, gaussians, cx, cy, ymin, ymax, xmin, xmax, N)
    return nothing
end

@inline function draw_kernel_inner(canvas, gaussians, cx, cy, ymin, ymax, xmin, xmax, N)
    _,H,W = size(canvas)

    if cx > W || cy > H
        return 0f0
    end

    density_per_unit_area = 50.0f0

    pixel_height = Float32(ymax-ymin)/H
    pixel_width = Float32(xmax-xmin)/W

    py = ymin + pixel_height * (cy-1)
    px = xmin + pixel_width * (cx-1)

    r,g,b = render_pixel(py,px,gaussians, N, density_per_unit_area)

    @inbounds canvas[1,cy,cx] = r
    @inbounds canvas[2,cy,cx] = g
    @inbounds canvas[3,cy,cx] = b

    # @inbounds loss = abs(target[1,cy,cx] - r) + abs(target[2,cy,cx] - g) + abs(target[3,cy,cx] - b)

    return
end

@inline function render_pixel(py,px,gaussians, N, density_per_unit_area)
    T = 1f0 # transmittance
    r = 0f0
    g = 0f0
    b = 0f0

    for G in 1:N
        # get position relative to gaussian
        x2 = px-gaussians[G_X,G]
        y2 = py-gaussians[G_Y,G]

        # now rotate
        x = x2*gaussians[G_COS_ANGLE,G] - y2*gaussians[G_SIN_ANGLE,G]
        y = x2*gaussians[G_SIN_ANGLE,G] + y2*gaussians[G_COS_ANGLE,G]

        density = exp(-((x/gaussians[G_SCALE_X,G])^2 + (y/gaussians[G_SCALE_Y,G])^2) / 2) / (6.2831855f0 * gaussians[G_SCALE_X,G] * gaussians[G_SCALE_Y,G]) * density_per_unit_area


        # alpha = clamp(gauss.opacity * Float32(density), 0f0, .999f0)
        alpha = min(gaussians[G_OPACITY,G] * density, .99f0)

        r += T * alpha * gaussians[G_R,G]
        g += T * alpha * gaussians[G_G,G]
        b += T * alpha * gaussians[G_B,G]

        T *= 1f0 - alpha
        # T < 0.01f0 && break # doesnt really impact perf in gpu case
    end
    return clamp(r, 0f0, 1f0), clamp(g, 0f0, 1f0), clamp(b, 0f0, 1f0)
end




end