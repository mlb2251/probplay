

module Gaussians

using CUDA
using Metal
using BenchmarkTools
using ForwardDiff
using Random

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

    # scale_x = rand()*5 * H/100
    Float32[
        rand()*H,
        rand()*W,
        rand()*5 * H/100,
        rand()*5 * H/100,
        # scale_x,
        # scale_x,
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

function test_gradients(;target=nothing, lr=300.0, lr_decay=.90, N = 100, iters = 20, check = false, mode = :reverse, device = :gpu, log_every=1)
    

    lr_decay = Float32(lr_decay)
    lr = Float32(lr)

    Random.seed!(0)

    if isnothing(target)
        H,W = 100,100
        target = zeros(Float32, 3, H, W)
        target[1,:,:] .= 1.
        # target[:,:,:] .= 0f0
    else
        target = Array(Float32.(channelview(load(target))))[1:3,:,:]
        _,H,W = size(target)    
    end
    html_body(html_img(Float64.(target), width="400px"))

    canvas = zeros(Float32, 3, H, W)
    transmittances = zeros(Float32, H, W)
    gaussians = rand_gauss(H,W,N)
    dgaussians = similar(gaussians)

    if device === :gpu
        CUDA.functional() && CUDA.device!(1)
        CUDA.functional() && CUDA.memory_status()
        converter = CUDA.functional() ? CuArray : MtlArray

        target = converter(target)
        canvas = converter(canvas)
        transmittances = converter(transmittances)
        gaussians = converter(gaussians)
        dgaussians = converter(dgaussians)
    end

    chunk_sz = min(H*W, 8)
    cfg = ForwardDiff.GradientConfig(nothing, gaussians, ForwardDiff.Chunk{chunk_sz}())
    canvas_dual = zeros(dual_type(Float32,cfg), 3, H, W)

    if device === :gpu
        canvas_dual = converter(canvas_dual)
    end

    println("Optimizing...")

    # dry run with lr=0 to precompile
    mode === :forward && grad_step_forward_mode(canvas_dual, target, gaussians, transmittances, 0f0, cfg)
    mode === :reverse && grad_step_reverse_mode(canvas, target, gaussians, transmittances, dgaussians, 0f0)

    @time for i in 1:iters
        @show i,lr
        if check
            gaussians_check = copy(gaussians)
        end

        if mode === :forward
            grad_step_forward_mode(canvas_dual, target, gaussians, transmittances, lr, cfg)
            if i % log_every == 0
                html_body(html_img(Float64.(ForwardDiff.value.(Array(canvas_dual))), width="400px"))
            end
        elseif mode === :reverse
            grad_step_reverse_mode(canvas, target, gaussians, transmittances, dgaussians, lr)
            if i % log_every == 0
                html_body(html_img(Float64.(Array(canvas)), width="400px"))
            end
        else
            error("unknown mode")
        end

        if check
            grad_step_forward_mode(canvas_dual, target, gaussians_check, transmittances, lr, cfg)
            check_grad(gaussians, gaussians_check)
        end

        # count = sum(gaussians[G_OPACITY,:] .< 0.001)
        # @show count/N

        lr *= lr_decay
    end

    # gaussians = converter(orig_gaussians)
    # target = converter(target)
    # canvas = converter(canvas)
    # cfg = ForwardDiff.GradientConfig(nothing, gaussians, ForwardDiff.Chunk{chunk_sz}())

    # lr = .01f0
    # grad_step(canvas, target, gaussians, 0f0, cfg)
    # @time for i in 1:10
    #     @show i
    #     grad_step(canvas, target, gaussians, lr, cfg)
    #     html_body(html_img(Float64.(ForwardDiff.value.(Array(canvas))), width="400px"))
    # end

    nothing
end

# using ReverseDiff

function check_grad(left,right)
    if !isapprox(left,right)
        println("ERR left != right")
        !isapprox(left[G_Y,:],right[G_Y,:]) && println("ERR G_Y")
        !isapprox(left[G_X,:],right[G_X,:]) && println("ERR G_X")
        !isapprox(left[G_SCALE_Y,:],right[G_SCALE_Y,:]) && println("ERR G_SCALE_Y")
        !isapprox(left[G_SCALE_X,:],right[G_SCALE_X,:]) && println("ERR G_SCALE_X")
        !isapprox(left[G_COS_ANGLE,:],right[G_COS_ANGLE,:]) && println("ERR G_COS_ANGLE")
        !isapprox(left[G_SIN_ANGLE,:],right[G_SIN_ANGLE,:]) && println("ERR G_SIN_ANGLE")
        !isapprox(left[G_OPACITY,:],right[G_OPACITY,:]) && println("ERR G_OPACITY: ", left[G_OPACITY,:], " vs ", right[G_OPACITY,:])
        !isapprox(left[G_R,:],right[G_R,:]) && println("ERR G_R: ", left[G_R,:], " vs ", right[G_R,:])
        !isapprox(left[G_G,:],right[G_G,:]) && println("ERR G_G: ", left[G_G,:], " vs ", right[G_G,:])
        !isapprox(left[G_B,:],right[G_B,:]) && println("ERR G_B: ", left[G_B,:], " vs ", right[G_B,:])
        println("")
    else
        println("check_grad correct!") # opacity is $(left[G_OPACITY,:]) vs $(right[G_OPACITY,:])")
    end
end


function grad_step_forward_mode(canvas_dual, target, gaussians, transmittances, lr, cfg)
    _,H,W = size(canvas_dual)
    dgaussians = ForwardDiff.gradient(gaussians, cfg) do gaussians
        draw_region(canvas_dual, gaussians, transmittances, 1, H, 1, W)
        sum(abs.(target .- canvas_dual)) / (H * W)
    end

    gaussians .-= lr .* dgaussians
    normalize_gaussians!(gaussians)
end

function grad_step_reverse_mode(canvas, target, gaussians, transmittances, dgaussians, lr)
    _,H,W = size(canvas)
    draw_region(canvas, gaussians, transmittances, 1, H, 1, W)
    dgaussians .= 0f0
    draw_region_backward(canvas, gaussians, transmittances, target, dgaussians, 1, H, 1, W)
    gaussians .-= lr .* dgaussians
    normalize_gaussians!(gaussians)
end







function normalize_gaussians!(gaussians::T) where T <: CuArray
    threads = 64
    blocks = cld(size(gaussians,2), threads)
    CUDA.@sync @cuda threads=threads blocks=blocks normalize_gaussians_kernel(gaussians)
end

function normalize_gaussians!(gaussians::T) where T <: MtlArray
    threads = 64
    blocks = cld(size(gaussians,2), threads)
    Metal.@sync @metal threads=threads groups=blocks normalize_gaussians_kernel(gaussians)
end

function normalize_gaussians_kernel(gaussians::T) where T <: CuDeviceArray
    G = (blockIdx().x-1) * blockDim().x + threadIdx().x
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

function normalize_gaussians!(gaussians::T) where T <: Array
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
    # gaussians[G_OPACITY,G] = max(gaussians[G_OPACITY,G], 0f0)

    # if scale actually drops to 0 you get divide by zero NaNs in the density calculation
    gaussians[G_SCALE_X,G] = max(gaussians[G_SCALE_X,G], 0.000001f0)
    gaussians[G_SCALE_Y,G] = max(gaussians[G_SCALE_Y,G], 0.000001f0)
    gaussians[G_R,G] = clamp(gaussians[G_R,G], 0f0, 1f0)
    gaussians[G_G,G] = clamp(gaussians[G_G,G], 0f0, 1f0)
    gaussians[G_B,G] = clamp(gaussians[G_B,G], 0f0, 1f0)
    return nothing
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

function draw_region(canvas::T, gaussians, transmittances, ymin, ymax, xmin, xmax) where T <: MtlArray
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))

    Metal.@sync begin
        @metal threads=threads groups=blocks draw_kernel(canvas, gaussians, transmittances, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
    return nothing
end

function draw_region(canvas::T, gaussians, transmittances, ymin, ymax, xmin, xmax) where T <: CuArray
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks draw_kernel(canvas, gaussians, transmittances, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
    return nothing
end

function draw_region(canvas::T, gaussians, transmittances, ymin, ymax, xmin, xmax) where T <: Array
    _,H,W = size(canvas)

    for cx in 1:W, cy in 1:H
        draw_kernel_inner(canvas, gaussians, transmittances, cx, cy, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
end


function draw_region_backward(canvas::T, gaussians, transmittances, target, dgaussians, ymin, ymax, xmin, xmax) where T <: MtlArray
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))

    Metal.@sync begin
        @metal threads=threads groups=blocks draw_kernel_backward(canvas, gaussians, transmittances, target, dgaussians,ymin, ymax, xmin, xmax, size(gaussians,2))
    end
    return nothing 
end

function draw_region_backward(canvas::T, gaussians, transmittances, target, dgaussians, ymin, ymax, xmin, xmax) where T <: CuArray
    C,H,W = size(canvas)
    threads = (16,16)
    blocks = (cld(W, threads[1]), cld(H, threads[2]))

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks draw_kernel_backward(canvas, gaussians, transmittances, target, dgaussians,ymin, ymax, xmin, xmax, size(gaussians,2))
    end
    return nothing
end


function draw_region_backward(canvas::T, gaussians, transmittances, target, dgaussians, ymin, ymax, xmin, xmax) where T <: Array
    _,H,W = size(canvas)

    for cx in 1:W, cy in 1:H
        draw_kernel_inner_backward(canvas, gaussians, transmittances, target, dgaussians, cx, cy, ymin, ymax, xmin, xmax, size(gaussians,2))
    end
end


function draw_kernel_backward(canvas::T, gaussians, transmittances, target, dgaussians,ymin, ymax, xmin, xmax, N) where T <: MtlDeviceArray
    # get which pixel of the canvas should this thread should draw
    cx,cy = Metal.thread_position_in_grid_2d()

    draw_kernel_inner_backward(canvas, gaussians, transmittances, target, dgaussians, cx, cy, ymin, ymax, xmin, xmax, N)
    return nothing
end

function draw_kernel_backward(canvas::T, gaussians, transmittances, target, dgaussians,ymin, ymax, xmin, xmax, N) where T <: CuDeviceArray
    # get which pixel of the canvas should this thread should draw
    cx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    draw_kernel_inner_backward(canvas, gaussians, transmittances, target, dgaussians, cx, cy, ymin, ymax, xmin, xmax, N)
    return nothing
end



function draw_kernel(canvas::T, gaussians, transmittances, ymin, ymax, xmin, xmax, N) where T <: CuDeviceArray
    # get which pixel of the canvas should this thread should draw
    cx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    cy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    draw_kernel_inner(canvas, gaussians, transmittances, cx, cy, ymin, ymax, xmin, xmax, N)
    return nothing
end

function draw_kernel(canvas::T, gaussians, transmittances, ymin, ymax, xmin, xmax, N) where T <: MtlDeviceArray
    # get which pixel of the canvas should this thread should draw
    cx,cy = Metal.thread_position_in_grid_2d()

    draw_kernel_inner(canvas, gaussians, transmittances, cx, cy, ymin, ymax, xmin, xmax, N)
    return nothing
end

@inline function draw_kernel_inner(canvas, gaussians, transmittances, cx, cy, ymin, ymax, xmin, xmax, N)
    _,H,W = size(canvas)

    if cx > W || cy > H
        return
    end

    pixel_height = Float32(ymax-ymin)/H
    pixel_width = Float32(xmax-xmin)/W

    py = ymin + pixel_height * (cy-1)
    px = xmin + pixel_width * (cx-1)

    r,g,b,T = render_pixel(py,px,gaussians, N)

    @inbounds canvas[1,cy,cx] = r
    @inbounds canvas[2,cy,cx] = g
    @inbounds canvas[3,cy,cx] = b
    @inbounds transmittances[cy,cx] = ForwardDiff.value(T)

    # @inbounds loss = abs(target[1,cy,cx] - r) + abs(target[2,cy,cx] - g) + abs(target[3,cy,cx] - b)

    return
end

const density_per_unit_area = 30f0

@inline function render_pixel(py, px, gaussians, N)
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

        exponent = -0.5f0 * ((x/gaussians[G_SCALE_X,G])^2 + (y/gaussians[G_SCALE_Y,G])^2)
        denominator = 6.2831855f0 * gaussians[G_SCALE_X,G] * gaussians[G_SCALE_Y,G]
        density = density_per_unit_area * exp(exponent) / denominator

        # we clamp density instead of alpha (which is what the paper does) because we dont want to lose the gradient to G_OPACITY
        # -- though the paper just doesnt factor the min() into the gradient anyways I believe
        clamped_density = min(density, .99f0)
        # @show clamped_density
        alpha = gaussians[G_OPACITY,G] * clamped_density
        # alpha = min(pre_alpha, .99f0)
        # if alpha < 1f0/255f0
        #     continue
        # end
        
        r = r + T * alpha * gaussians[G_R,G]
        g = g + T * alpha * gaussians[G_G,G]
        b = b + T * alpha * gaussians[G_B,G]

        T = T * (1f0 - alpha)

        # T < 0.01f0 && break # doesnt really impact perf in gpu case
    end
    return r, g, b, T
end

function draw_kernel_inner_backward(canvas, gaussians, transmittances, target, dgaussians, cx, cy, ymin, ymax, xmin, xmax, N) # ,::Val{N}) where N
    # START boilerplate from draw_kernel_inner
    _,H,W = size(canvas)

    # Metal.synchronize_threads()
    # dgauss = MtlThreadGroupArray(Float32, G_PARAMS)
    # Metal.synchronize_threads()
    # gauss .= gaussians

    if cx > W || cy > H
        return
    end

    pixel_height = Float32(ymax-ymin)/H
    pixel_width = Float32(xmax-xmin)/W

    py = ymin + pixel_height * (cy-1)
    px = xmin + pixel_width * (cx-1)

    # END boilerplate from draw_kernel_inner

    # go backwards through gaussians, this time increasing the transmittance as we go
    T_after = transmittances[cy,cx]
    r_after = canvas[1,cy,cx]
    g_after = canvas[2,cy,cx]
    b_after = canvas[3,cy,cx]
    r_acc = g_acc = b_acc = 0f0

    r_loss_signed = target[1,cy,cx] - r_after
    g_loss_signed = target[2,cy,cx] - g_after
    b_loss_signed = target[3,cy,cx] - b_after
    
    # loss = abs(r_loss_signed) + abs(g_loss_signed) + abs(b_loss_signed)

    # here `dr` is the loss with respect to any of the `r` values constructed during the loop
    # dloss_dr = dloss_d_r_loss_signed * dr_loss_signed_d_r_final * dr_final_d_r
    # dloss_dr = -1f0 * (r_loss_signed > 0f0 ? 1f0 : -1f0) * 1f0

    # to align with how ForwardDiff.derivative(abs, 0f0) == 1f0 
    # however this does cause weird instabilities 
    
    dloss_dr = (r_loss_signed >= 0f0 ? -1f0 : 1f0) / (H*W)
    dloss_dg = (g_loss_signed >= 0f0 ? -1f0 : 1f0) / (H*W)
    dloss_db = (b_loss_signed >= 0f0 ? -1f0 : 1f0) / (H*W)

    # old version that sets deriv to 0f0 at loss of 0f0
    # dloss_dr = r_loss_signed == 0f0 ? 0f0 : (r_loss_signed > 0f0 ? -1f0 : 1f0) / (H*W)
    # dloss_dg = g_loss_signed == 0f0 ? 0f0 : (g_loss_signed > 0f0 ? -1f0 : 1f0) / (H*W)
    # dloss_db = b_loss_signed == 0f0 ? 0f0 : (b_loss_signed > 0f0 ? -1f0 : 1f0) / (H*W)

    # dloss_dr != 0f0 && @show dloss_dr

    # gauss = zeros(Float32, G_PARAMS)

    for G in reverse(1:N)

        # Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        # if thread_position_in_grid_1d() == 1
            # dgauss[G_OPACITY] = 0f0
            # dgauss[G_R] = 0f0
            # dgauss[G_G] = 0f0
            # dgauss[G_B] = 0f0
            # dgauss[G_X] = 0f0
            # dgauss[G_Y] = 0f0
            # dgauss[G_SCALE_X] = 0f0
            # dgauss[G_SCALE_Y] = 0f0
            # dgauss[G_COS_ANGLE] = 0f0
            # dgauss[G_SIN_ANGLE] = 0f0
        # end

        # Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        # thread = thread_position_in_threadgroup_1d()
        # if thread < G_PARAMS
        #     gauss[thread] = gaussians[thread,G]
        # end

        # if threads_per_threadgroup_1d() < G_PARAMS && thread == 1
        #     for thread in threads_per_threadgroup_1d():G_PARAMS
        #         gauss[thread] = gaussians[thread,G]
        #     end
        # end


        # gauss[G_X] = gaussians[G_X,G]

        # START we repeat all this alpha calculation as the 3D gaussian splatting paper does, instead of storing the per-gauss per-pixel results
        x2 = px-gaussians[G_X,G]
        y2 = py-gaussians[G_Y,G]

        dx2_dG_X = -1f0
        dy2_dG_Y = -1f0

        x = x2*gaussians[G_COS_ANGLE,G] - y2*gaussians[G_SIN_ANGLE,G]
        y = x2*gaussians[G_SIN_ANGLE,G] + y2*gaussians[G_COS_ANGLE,G]

        # normed_cos_angle = gaussians[G_COS_ANGLE,G] / sqrt(gaussians[G_COS_ANGLE,G]^2 + gaussians[G_SIN_ANGLE,G]^2)
        # = gaussians[G_COS_ANGLE,G] * (gaussians[G_COS_ANGLE,G]^2 + gaussians[G_SIN_ANGLE,G]^2) ^ -0.5f0
        # dnormed_cos_angle_dcos_angle = (norm) ^ -0.5f0 + gaussians[G_COS_ANGLE,G] * -0.5f0 * (norm) ^ -1.5f0 * 2 * gaussians[G_COS_ANGLE,G]
        # norm = gaussians[G_COS_ANGLE,G]^2 + gaussians[G_SIN_ANGLE,G]^2
        # dnormed_cos_angle_dcos_angle = 1/sqrt(norm) + gaussians[G_COS_ANGLE,G] * -0.5f0  *
        # todo do it on paper... also unclear this is the way to go lol

        dx_dx2 = gaussians[G_COS_ANGLE,G]
        dx_dy2 = -gaussians[G_SIN_ANGLE,G]
        dy_dx2 = gaussians[G_SIN_ANGLE,G]
        dy_dy2 = gaussians[G_COS_ANGLE,G]

        dx_dG_COS_ANGLE = x2
        dx_dG_SIN_ANGLE = -y2
        dy_dG_COS_ANGLE = y2
        dy_dG_SIN_ANGLE = x2

        exponent = -0.5f0 * ((x/gaussians[G_SCALE_X,G])^2 + (y/gaussians[G_SCALE_Y,G])^2)
        denominator = 6.2831855f0 * gaussians[G_SCALE_X,G] * gaussians[G_SCALE_Y,G]
        density = density_per_unit_area * exp(exponent) / denominator

        clamped_density = min(density, .99f0)
        alpha = gaussians[G_OPACITY,G] * clamped_density


        # dexponent_dG_SCALE_X = -0.5f0 * 2f0 * (x / gaussians[G_SCALE_X,G]) * (- x / gaussians[G_SCALE_X,G]^2)
        # ... simplifies to:
        dexponent_dG_SCALE_X = x^2 / gaussians[G_SCALE_X,G]^3
        dexponent_dG_SCALE_Y = y^2 / gaussians[G_SCALE_Y,G]^3

        # dexponent_dx = -0.5f0 * 2f0 * (x / gaussians[G_SCALE_X,G]^2)
        # ... simplifies to:
        dexponent_dx = - x / gaussians[G_SCALE_X,G]^2
        dexponent_dy = - y / gaussians[G_SCALE_Y,G]^2

        ddenominator_dG_SCALE_X = 6.2831855f0 * gaussians[G_SCALE_Y,G]
        ddenominator_dG_SCALE_Y = 6.2831855f0 * gaussians[G_SCALE_X,G]

        ddensity_dexponent = density # wild
        # ddensity_ddenominator = density_per_unit_area * exp(exponent) *  - (1/denominator^2)
        # ... simplifies to:
        ddensity_ddenominator = -density / denominator

        # this is what forwarddiff would do, but it seems possibly undesirable!
        dclamped_density_ddensity = density < .99f0 ? 1f0 : 0f0
        # dclamped_density_ddensity = 1f0

        
        dalpha_dG_OPACITY = clamped_density
        dalpha_dclamped_density = gaussians[G_OPACITY,G]

        # if alpha < 1f0/255f0
        #     continue
        # end

        # END alpha calculation

        # now to get the T value used at this point in the computation we actually need to step it backwards
        # before using it
        T_before = T_after/(1f0 - alpha)

        # this is the most complex of the equations. note r_acc is the sum of all the r values of the gaussians BEHIND this one
        # so it makes sense that increasing alpha would decrease the contribution of the gaussians behind it. Note that sum
        # could probably also be calculated by subtracting the current-ish r value from the total r value but this is how
        # im writing it at first at least. I'll write the full derivation of this all somewhere

        # Intuitively: increasing r will have 2 effects which are the two terms here. First, itll increase the contribution of this
        # gaussian to the pixel (proportional to the current transmittance and the R value of the gaussian). Second, itll decrease
        # the contribution of all gaussians behind this one, so that effect will be proportional to the accumulated r value of all
        # those gaussians, and will also be inversely proportional to 1-alpha. The latter means that in the limit of alpha=1 (total occlusion)
        # a tiny decrease in alpha will result in an infinite increase in the contribution of the gaussians behind it – which makes sense since
        # they had near zero impact before.

        # Derivation, starting from the definition of the final color `r`
        # T_before[i] = product(1-alpha[j] for j in 1:i-1)
        # r = sum(gaussians[G_R,i] * alpha[i] * T_before[i] for i in 1:N)
        # split up the sum into 3 chunks: 1:G-1, G, and G+1:N
        # r = sum(gaussians[G_R,i] * alpha[i] * T_before[i] for i in 1:G-1) + (gaussians[G_R,G] * alpha[G] * T_before[G]) + sum(gaussians[G_R,i] * alpha[i] * T_before[i] for i in G+1:N)
        # now lets do d/dalpha[G].
        # First term: Note that T_before[i] depends on all alpha[<i] so the first sum has no dependence and will be zero
        # Second term: easy, and note T_before[G] has no dependence on alpha[G], so the second term is gaussians[G_R,G] * T_before[G]
        # Third term: note alpha[i] will never be alpha[G] so the only dependence is from T_before[i] where i>G always so there is always a factor of (1-alpha[G]) in that product which is the only dependence.
        # We can rewrite T_before[i] as  (T_before[i] / (1-alpha[k])) * (1-alpha[k])
        # Note that the derivative of the first term (T_before[i] / (1-alpha[k])) is zero since there is no alpha[k] in that term after you divide it out
        # so dT_before[i]/dalpha[k] = (T_before[i] / (1-alpha[k])) * -1
        # so the derivative of the ith term in our sum becomes (T_before[i] / (1-alpha[k])) * -1 * gaussians[G_R,i] * alpha[i]
        # which simplifies to -gaussians[G_R,i] * alpha[i] * T_before[i] / (1-alpha[k]))
        # and we can pull that -(1-alpha[k])) out of the sum so our overall sum derivative becomes
        # - sum(gaussians[G_R,i] * alpha[i] * T_before[i] for i in G+1:N) / (1-alpha[k])


        dr_dalpha = T_before * gaussians[G_R,G] - r_acc/(1-alpha)
        dg_dalpha = T_before * gaussians[G_G,G] - g_acc/(1-alpha)
        db_dalpha = T_before * gaussians[G_B,G] - b_acc/(1-alpha)

        # G_R G_G G_B gradients are all easy
        dr_dG_R = dg_dG_G = db_dG_B = T_before * alpha

        # update those gradients!
        atomic_add_generic!(dgaussians, G_R, G, dloss_dr * dr_dG_R)
        atomic_add_generic!(dgaussians, G_G, G, dloss_dg * dg_dG_G)
        atomic_add_generic!(dgaussians, G_B, G, dloss_db * db_dG_B)

        # helpers
        dloss_dalpha = dloss_dr * dr_dalpha + dloss_dg * dg_dalpha + dloss_db * db_dalpha
        dloss_ddensity = dloss_dalpha * dalpha_dclamped_density * dclamped_density_ddensity
        dloss_ddenominator = dloss_ddensity * ddensity_ddenominator
        dloss_dexponent = dloss_ddensity * ddensity_dexponent
        dloss_dx = dloss_dexponent * dexponent_dx
        dloss_dy = dloss_dexponent * dexponent_dy
        dloss_dx2 = dloss_dx * dx_dx2 + dloss_dy * dy_dx2
        dloss_dy2 = dloss_dx * dx_dy2 + dloss_dy * dy_dy2

        atomic_add_generic!(dgaussians, G_OPACITY, G, dloss_dalpha * dalpha_dG_OPACITY)

        atomic_add_generic!(dgaussians, G_SCALE_X, G, dloss_dexponent * dexponent_dG_SCALE_X + dloss_ddenominator * ddenominator_dG_SCALE_X)
        atomic_add_generic!(dgaussians, G_SCALE_Y, G, dloss_dexponent * dexponent_dG_SCALE_Y + dloss_ddenominator * ddenominator_dG_SCALE_Y)

        atomic_add_generic!(dgaussians, G_COS_ANGLE, G, dloss_dx * dx_dG_COS_ANGLE + dloss_dy * dy_dG_COS_ANGLE)
        atomic_add_generic!(dgaussians, G_SIN_ANGLE, G, dloss_dx * dx_dG_SIN_ANGLE + dloss_dy * dy_dG_SIN_ANGLE)

        atomic_add_generic!(dgaussians, G_X, G, dloss_dx2 * dx2_dG_X)
        atomic_add_generic!(dgaussians, G_Y, G, dloss_dy2 * dy2_dG_Y)

        # how much this gaussian will contribute to the pixel
        r_diff = T_before * alpha * gaussians[G_R,G]
        g_diff = T_before * alpha * gaussians[G_G,G]
        b_diff = T_before * alpha * gaussians[G_B,G]

        # accumulate how much we've contributed to the pixel in our reverse traversal
        r_acc += r_diff
        g_acc += g_diff
        b_acc += b_diff

        r_after -= r_diff
        g_after -= g_diff
        b_after -= b_diff
        T_after = T_before

        # if thread_position_in_grid_1d() == 1
            # dgaussians[G_OPACITY,G] += dgauss[G_OPACITY]
            # Metal.@atomic dgaussians[G_R,G] += dgauss[G_R]
            # Metal.@atomic dgaussians[G_G,G] += dgauss[G_G]
            # Metal.@atomic dgaussians[G_B,G] += dgauss[G_B]
            # Metal.@atomic dgaussians[G_X,G] += dgauss[G_X]
            # Metal.@atomic dgaussians[G_Y,G] += dgauss[G_Y]
            # Metal.@atomic dgaussians[G_SCALE_X,G] += dgauss[G_SCALE_X]
            # Metal.@atomic dgaussians[G_SCALE_Y,G] += dgauss[G_SCALE_Y]
            # Metal.@atomic dgaussians[G_COS_ANGLE,G] += dgauss[G_COS_ANGLE]
            # Metal.@atomic dgaussians[G_SIN_ANGLE,G] += dgauss[G_SIN_ANGLE]
        # end
    end


end


@inline function atomic_add_generic!(dgaussians::T, param, G, val) where T <: CuDeviceArray 
    CUDA.@atomic dgaussians[param,G] += val
end

@inline function atomic_add_generic!(dgaussians::T, param, G, val) where T <: MtlDeviceArray 
    Metal.@atomic dgaussians[param,G] += val
    # Metal.@atomic dgaussians[param] += val
    # dgaussians[param,G] += val
    # Metal.atomic_fetch_add_explicit(Metal.pointer(dgaussians,param), val)
    # Metal.atomic_fetch_add_explicit(Metal.pointer(dgaussians,10*G+param), val)
end

@inline function atomic_add_generic!(dgaussians::T, param, G, val) where T <: Array 
    dgaussians[param,G] += val
end



end