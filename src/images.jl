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
    (H,W,T) = get_args(trace)
    @assert t <= T
    sprites = Sprite[]
    for i in 1:trace[:init => :num_sprite_types]

        color = trace[:init => :init_sprites => i => :color] #?
        
        mask = trace[:init => :init_sprites => i => :mask]

        sprite_type = Sprite(mask, color)
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

function img_diff(img1, img2)
    @. (img1 - img2) / 2. + 0.5
end
function img_diff_sum(img1, img2)
    sum(abs.(img_diff(img1, img2)))
end


function render_heatmap(heatmap)
    (H, W) = size(heatmap)


    max = maximum(heatmap)
    min = minimum(heatmap)
    #@show min, max
    #@show heatmap

    #@show heatmap[10, :]
    normalized_heatmap = (heatmap .- min) ./ (max - min)
    #@show normalized_heatmap[10, :]

    #spread_heatmap = log.(normalized_heatmap .+ 1)
    #heatmap to power of 10
    spread_heatmap = normalized_heatmap .^ 50
    #spread_heatmap = e .^ normalized_heatmap
    final_heatmap = spread_heatmap

    
    render_matrix(final_heatmap, 1)

end 

function get_other_colors(color)
    if color == 1
        return [2,3]
    elseif color == 2
        return [1,3]
    elseif color == 3
        return [1,2]
    end
end


function render_matrix(m, color=2, render=false)
    #getting between 0 and 1
    m = (m .- minimum(m)) ./ (maximum(m) - minimum(m))
    H, W = size(m)
    img = zeros(3, H, W)
    for i in 1:H
        for j in 1:W

            img[color, i, j] = 1
            other_colors = get_other_colors(color)

            img[other_colors[1], i, j] = m[i,j]
            img[other_colors[2], i, j] = m[i,j]
        end
    end
    if render
        html_body(html_img(img))
        render()
    else
        return html_img(img)
    end 
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
    #@show frames
    max = maximum([maximum(frame) for frame in frames])
    res = []
    for frame in frames
        colored = [RGB(HSV(px/max*360, .8, .7)) for px in frame]
        colored[frame .== 0] .= RGB(0,0,0)
        orig !== nothing && (colored = vcat(colorview(RGB,orig), colored))
        push!(res,colored)
    end
    res
end

"""
get N colors evenly distributed in terms of hue
"""
function get_colors(N)
    rgbs = [RGB(HSV(i/N*360, .8, .8)) for i in 1:N]
    return [[c.r, c.g, c.b] for c in rgbs]
end

function obj_frame(objs, sprites, H, W)
    """
    fills a frame of each pixel which object it belongs to
    """
    frame = zeros(Int, H, W)
    objnum = 0 
    for obj in objs
        objnum += 1 
        sprite = sprites[obj.sprite_index]
        # @show size(sprite.mask)
        # print("did w emake it here")
        for i in 1:size(sprite.mask)[1]
            for j in 1:size(sprite.mask)[2]
                #@show i,j
                if sprite.mask[i,j] == 1
                    itofill = i+obj.pos.y-1
                    jtofill = j+obj.pos.x-1
                    if 0 < itofill <= H && 0 < jtofill <= W
                        frame[i+obj.pos.y-1, j+obj.pos.x-1] = objnum#be careful for off by one
                    end
                    
                end 
            end
        end
    end	
    frame 
end 

function sprite_frame(objs, sprites, H, W)
    """
    fills a frame of each pixel which sprite ind it belongs to
    """
    frame = zeros(Int, H, W)
    objnum = 0 
    for obj in objs
        objnum += 1 
        sprite = sprites[obj.sprite_index]
        for i in 1:size(sprite.mask)[1]
            for j in 1:size(sprite.mask)[2]
                #@show i,j
                if sprite.mask[i,j] == 1
                    itofill = i+obj.pos.y-1
                    jtofill = j+obj.pos.x-1
                    if 0 < itofill <= H && 0 < jtofill <= W
                        frame[i+obj.pos.y-1, j+obj.pos.x-1] = obj.sprite_index#be careful for off by one
                    end
                    
                end 
            end
        end
    end	
    frame 
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
