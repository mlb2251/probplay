import DynamicForwardDiff: Dual
import GenTraceKernelDSL: TraceToken


# workaround for annoying bug where if you try to access the return value for trace[:init] followed by trace[:init => :init_state => :N] has_key will get confused
# because itll think that :init is both a submap and a value and it'll crash.
env_of_trace(trace::TraceToken) = env_of_trace(trace.trace)
state_of_trace(trace::TraceToken, t) = state_of_trace(trace.trace, t)

function env_of_trace(trace)
    return trace[:init]
end

function state_of_trace(trace, t)
    if t == 0
        return trace[:init => :init_state]
    else
        return trace[:steps => t]
    end
end


# function objs_from_trace(trace, t)
#     (H,W,T) = get_args(trace)
#     @assert t <= T
#     objs = Object[] #why grey text here?
#     for i in 1:trace[:init => :init_state => :N]
#         if t == 0
#             pos = trace[:init => :init_state => :init_objs => i => :pos]
#         else
#             pos = trace[:steps => t => :objs => i => :pos]
#         end
        
#         sprite_index = trace[:init => :init_state => :init_objs => i => :sprite_index]
#         #@show sprite_index
#         obj = Object(sprite_index, pos)
#         push!(objs, obj)
#     end
#     objs
# end

# function sprites_from_trace(trace, t)
#     (H,W,T) = get_args(trace)
#     @assert t == 0 # we havent implemented sprites changing over time yet
#     sprites = Sprite[]
#     for i in 1:trace[:init => :num_sprites]

#         color = trace[:init => :init_sprites => i => :color] #?
        
#         shape = trace[:init => :init_sprites => i => :shape]

#         if color isa Vector{Dual{Nothing, Float64}}
#             sprite = Sprite(shape, [color[1].value, color[2].value, color[3].value])
#         else
#             sprite = Sprite(shape, color)
#         end

#         push!(sprites, sprite)
#     end
#     sprites
# end

function render_trace_frame(trace, t)
    (H,W,T) = get_args(trace)
    draw(H, W, state_of_trace(trace,t).objs, env_of_trace(trace).sprites)
end

function render_trace(trace)
    (H,W,T) = get_args(trace)
    stack([render_trace_frame(trace, t) for t in 0:T-1])
end

function img_diff(img1, img2)
    @. (img1 - img2) / 2. + 0.5
end
function img_diff_sum(img1, img2)
    sum(abs.(img_diff(img1, img2)))
end







function inflate(frame, scale=4)
    repeat(frame, inner=(scale,scale))
end

"""
Useful for visualizing the output of process_first_frame() etc

takes an HW frame of integers and returns a version with a unique
RGB color for each integer. If the original CHW frame orig is provided, it will
be concatenated onto to the result.
"""
function color_labels(frames...; orig=nothing)
    max = maximum([maximum(frame) for frame in frames])
    res = []
    for frame in frames
        colored = [RGB{Float64}(HSV(px/max*360, .8, .7)) for px in frame]
        colored[frame .== 0] .= RGB(0,0,0)
        orig !== nothing && (colored = vcat(colorview(RGB,orig), colored))
        push!(res,colored)
    end
    res
end

function games()
    [x for x in readdir("out/gameplay") if occursin("v5",x)]
end

function frames(game)
    frames = length(["out/gameplay/$game/$x" for x in readdir("out/gameplay/$game") if occursin("png",x)])
    ["out/gameplay/$game/$i.png" for i in 1:frames]
end

"""
Loads and properly sorts all frames of gameplay in a directory,
assumes names like 10.png 100.png.
Returns a (C, H, W, T) array of framges
"""
function load_frames(path)
    files = readdir(path)
    sort!(files, by = f -> parse(Int, split(f, ".")[1]))
    stack([Float64.(channelview(load(joinpath(path, f)))) for f in files], dims=4)
end

"""
crops the specified amounts off of the top, bottom, left, and right of a (C, H, W, T) array of images
"""
function crop(img; top=1, bottom=0, left=1, right=0, tstart=0, tend=0, tskip=1)
    img[:, top:end-bottom, left:end-right, 1+tstart:tskip:end-tend]
end
