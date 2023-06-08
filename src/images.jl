using TestImages
using Plots
using Distributions
using Images


function test()
    input = [[3,2,4,2,7,6] [8,0,2,1,7,8] [2,2,10,4,1,9] [1,5,4,6,5,0] [5,4,1,7,5,6] [5,0,2,7,6,8]]
    filter = [[1,1,1] [0,0,0] [-1,-1,-1]]

    input = Float32.(input)
    filter = Float32.(filter)

    convolve(input, filter)

end

function norm_filter(filter)
    norm = sum(filter[filter .> 0])
    norm > 0 || error("filter must have some positive values")
    filter ./ norm
end

function load_img(path)
    img = load(path)
    img = Float64.(channelview(img))
    R = img[1,:,:]
    G = img[2,:,:]
    B = img[3,:,:]
    (R,G,B)
end

function load_testimg(name)
    img = Float64.(channelview(testimage(name)))
    R = img[1,:,:]
    G = img[2,:,:]
    B = img[3,:,:]
    (R,G,B)
end

# "something proportional to a 2d gaussian"
# function gaussian(x, y, mu_x, mu_y, cov)

# end

"""
Constructs a filter of size `size*size` that is proportional to the
pdf of the given distribution at the center of each cell, but normalized to sum to 1.
"""
function dist_filter(dist, size)
    filter = zeros(size,size)
    cell_size = (1 / size)
    for i in 1:size, j in 1:size
        filter[i,j] = pdf(dist, [(i - .5) * cell_size - 0.5, (j - .5) * cell_size - 0.5])
    end
    filter ./ sum(filter) 
end

function show_filter(filter)
    display(filter)
    heatmap(reverse(filter,dims=1))
end

mutable struct Filter
    dist::Distribution
    size::Union{Int,Nothing}

    Filter(dist::Distribution, size::Int) = new(dist, size)
    Filter(dist::Distribution) = new(dist, nothing)
end

struct ScaledNormal
    dist::MvNormal
    scale::Float64

    ScaledNormal(dist::MvNormal, scale::Float64) = new(dist, scale)
    ScaledNormal(scale::Float64) = new(MvNormal([0,0],.5), scale)
    ScaledNormal() = ScaledNormal(1.0)
end

(-)(d::ScaledNormal) = ScaledNormal(d.dist, -d.scale)
(*)(d::ScaledNormal, x) = ScaledNormal(d.dist, d.scale * x)

function rot(d::ScaledNormal, deg)
    ang = -deg * pi / 180
    rot_matx = [cos(ang) -sin(ang); sin(ang) cos(ang)]
    ScaledNormal(rot_matx * d.dist, d.scale)
end

function (f::Filter)(img; size=nothing)

end

function test3()

    # filter = dist_filter(MvNormal([0,-.33], [.2,.2]), 5) - dist_filter(MvNormal([0,.33], [.2,.2]), 5)
    filter = dist_filter(MvNormal([-0.5,0.5], [.2,.2]), 3) - dist_filter(MvNormal([0.5,-0.5], [.2,.2]), 3)

    display(filter)
    heatmap(reverse(filter,dims=1))

    # heatmap(reverse(dist_filter(MvNormal([-0.5,-.05], [.25,1]), 7),dims=1))
end

"2x2 rotation matrix; clockwise; degrees"
function rot(deg)
    ang = -deg * pi / 180
    [cos(ang) -sin(ang); sin(ang) cos(ang)]
end

scale(x,y) = [x 0; 0 y]
scale(x) = scale(x,x)

"add to a matrix to translate by some distance in a direction; 0 is right"
function trans(distance,deg)
    ang = deg * pi / 180
    [sin(ang) * distance; cos(ang) * distance]
end

function normal()
    MvNormal([0,0],.5)
end


function antisym(dist, distance, deg)
    (trans(distance,deg) + dist), (trans(distance,deg+180) + dist)
end

function test2()

    # (R,G,B) = load_img("out/gameplay/ALE-Carnival-v5/1.png")
    # (R,G,B) = load_testimg("mandrill")
    (R,G,B) = load_img("data/test_image.png")

    blur(img) = convolve(img, dist_filter(MvNormal([0,0], .25), 3))

    (R,G,B) = (blur(R), blur(G), blur(B))

    # filter_horiz = [[-1,0,1] [-2,0,2] [-1,0,1]] # [-1;0;1 ;; -2;0;2 ;; -1;0;1]
    # filter_vert = transpose(filter_horiz)


    d = scale(.4) * normal()
    filter_vert = dist_filter(trans(.33,180) + d, 7) - dist_filter(trans(.33,0) + d, 7)
    filter_horiz = transpose(filter_vert)

    # filter_diag = [1;1;0 ;; 1;0;-1 ;; 0;-1;-1]
    filter_diag = dist_filter(MvNormal([-0.33,-0.33], .2), 3) - dist_filter(MvNormal([0.33,0.33], .2), 3)

    diag = convolve(R, norm_filter(filter_diag))
    horiz = convolve(R, norm_filter(filter_horiz))
    vert = convolve(R, norm_filter(filter_vert))

    filter_top_right = dist_filter(MvNormal([-0.5,0.5], [.2,.2]), 3) - dist_filter(MvNormal([0.5,-0.5], [.2,.2]), 3)
    filter_bottom_left = transpose(filter_top_right)
    filter_middle = dist_filter(MvNormal([0,0], [.2,.2]), 3)


    res = (
            convolve(diag, norm_filter(filter_middle))
        .* convolve(horiz, norm_filter(filter_top_right))
        .* convolve(vert, norm_filter(filter_bottom_left))
    )

    # res = (diag .* horiz .* vert) 
    # res = diag ./ 3

    
    # Gray.(res)
    heatmap(reverse(res,dims=1))
    # colorview(RGB,R,G,B)



    # res = convolve(img,filter)

    # filter2 = transpose(filter)
    # res2 = convolve(img,filter2)

    # R = convolve(img[1,:,:],filter)
    # G = convolve(img[2,:,:],filter)
    # B = convolve(img[3,:,:],filter)

    # res = colorview(RGB,R,G,B)
    # res = colorview(Gray,res)
    # res2 = colorview(Gray,res2)

    # @show maximum(res)

    # colorview(RGB,convolve(Float64.(channelview(img))[1,:,:],f))
    # mosaicview([orig_img,res,res2, res+res2], nrow=1)
end




function convolve(input, filter)

    filter_sz = size(filter, 1)
    ndims(input) == 2 || error("input must be 2D")
    ndims(filter) == 2 && size(filter, 1) == size(filter, 2) || error("filter must be square")    
    filter_sz % 2 == 1 || error("filter must have odd dimensions")

    # pad
    padding = (filter_sz - 1) ÷ 2
    padded_input = zeros(size(input) .+ 2*padding)
    padded_input[padding+1:end-padding, padding+1:end-padding] = input

    # convolve
    res = similar(input)
    for I in CartesianIndices(res)
        i, j = Tuple(I)
        res[I] = 0
        for i2 in 1:filter_sz, j2 in 1:filter_sz
            res[I] += padded_input[i+i2-1, j+j2-1] * filter[i2, j2]
        end
    end

    res

end
