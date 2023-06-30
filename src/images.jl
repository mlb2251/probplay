function objs_from_trace(trace, t)
    (H,W,T) = get_args(trace)
    @assert t <= T
    objs = Object[] #why grey text here?
    for i in 1:trace[:init => :N]
        if t == 0
            pos = trace[:init => :init_objs => i => :pos]
        else
            pos = trace[:steps => t => :objs => i => :pos]
        end
        
        sprite_index = trace[:init => :init_objs => i => :sprite_index]
        #@show sprite_index
        obj = Object(sprite_index, pos)
        push!(objs, obj)
    end
    objs
end

function sprites_from_trace(trace, t)
    NUM_SPRITE_TYPES = 4
    (H,W,T) = get_args(trace)
    @assert t <= T
    sprites = Sprite_Type[]
    for i in 1:NUM_SPRITE_TYPES

        color = trace[:init => :init_sprites => i => :color] #?
        
        shape = trace[:init => :init_sprites => i => :shape]

        sprite_type = Sprite_Type(shape, color)
        push!(sprites, sprite_type)
    end
    sprites
end

function render_trace_frame(trace, t)
    (H,W,T) = get_args(trace)
    draw(H, W, objs_from_trace(trace,t), sprites_from_trace(trace,t))
end

function render_trace(trace)
    (H,W,T) = get_args(trace)
    stack([render_trace_frame(trace, t) for t in 0:T-1])
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
function color_labels(frame, orig=nothing)
    max = maximum(frame)
    colored = [RGB(HSV(px/max*360, .8, .7)) for px in frame]
    colored[frame .== 0] .= RGB(0,0,0)
    
    orig !== nothing && (colored = vcat(colorview(RGB,orig), colored))
    colored
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
function crop(img; top=0, bottom=0, left=0, right=0, tstart=0, tend=0, tskip=1)
    img[:, top:end-bottom, left:end-right, 1+tstart:tskip:end-tend]
end
