
using Gen
using LinearAlgebra
using Images
using Distributions
using Plots
# using OffsetArrays


struct Position
    x::Int
    y::Int
end

struct UniformPosition <: Gen.Distribution{Position} end

function Gen.random(::UniformPosition, height, width)
    Position(rand(1:height), rand(1:width))
end

function Gen.logpdf(::UniformPosition, pos, height, width)
    if pos.x < 1 || pos.x > height || pos.y < 1 || pos.y > width
        return -Inf
    else
        # uniform distribution over height*width positions
        return -log(height*width)
    end
end

const uniform_position = UniformPosition()

(::UniformPosition)(h, w) = random(UniformPosition(), h, w)


struct UniformDriftPosition <: Gen.Distribution{Position} end

function Gen.random(::UniformDriftPosition, pos, max_drift)
    Position(pos.x + rand(-max_drift:max_drift),
             pos.y + rand(-max_drift:max_drift))
end

function Gen.logpdf(::UniformDriftPosition, pos_new, pos, max_drift)
    # discrete uniform over square with side length 2*max_drift + 1
    return -2*log(2*max_drift + 1)
end

const uniform_drift_position = UniformDriftPosition()

(::UniformDriftPosition)(pos, max_drift) = random(UniformDriftPosition(), pos, max_drift)



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
    pos::Position
end


function canvas(height=210, width=160, background=RGB{Float64}(0,0,0))
    fill(background, height, width)
end

function draw!(canvas, obj::Object)
    sprite = obj.sprite
    for I in CartesianIndices(sprite.mask)
        i, j = Tuple(I)
        if sprite.mask[i,j]
            offy = obj.pos.y+i-1
            offx = obj.pos.x+j-1
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


@gen function model(canvas_height, canvas_width, T)

    var = .1

    N ~ poisson(5)
    objs = Object[]

    # initialize objects
    for i in 1:N
        w = {(i => :width)} ~ uniform_discrete(1,100)
        h = {(i => :height)} ~ uniform_discrete(1,100)
        shape = {(i => :shape)} ~ bernoulli_2d(0.5, h,w)
        color = {(i => :color)} ~ rgb_dist()
        sprite = Sprite(shape, color)

        pos = {(1 => i => :pos)} ~ uniform_position(canvas_height, canvas_width)

        obj = Object(sprite, pos)
        push!(objs, obj)
    end

    # render
    rendered = Array(channelview(draw!(canvas(canvas_height, canvas_width), objs)))
    observed_image = {1 => :observed_image} ~ image_likelihood(rendered, var)

    for t in 2:T
        for i in 1:N
            pos = {t => i => :pos} ~ uniform_drift_position(objs[i].pos, 3)
            objs[i] = Object(objs[i].sprite, pos)
        end
        rendered = Array(channelview(draw!(canvas(canvas_height, canvas_width), objs)))
        observed_image = {t => :observed_image} ~ image_likelihood(rendered, var)
    end

    return
end

function inflate(frame, scale=4)
    repeat(frame, inner=(scale,scale))
end

"""
takes an HW frame of integers and returns a version with a unique
RGB color for each integer. If the original CHW frame orig is provided, it will
be concatenated onto to the result.
"""
function color_labels(frame, orig=nothing)
    max = maximum(frame)
    colored = [RGB(HSV(h/max*360, .8, .7)) for h in frame]
    orig !== nothing && (colored = vcat(colorview(RGB,orig), colored))
    inflate(colored)
end

# frame = crop(load_frames("out/benchmarks/frostbite_1"), top=120, bottom=25, left=20)[:,:,:,20]
# color_labels(process_first_frame(frame),frame)
function process_first_frame(frame, threshold=.05)
    (C, H, W) = size(frame)

    # contiguous color
    cluster = zeros(Int, H, W)
    curr_cluster = 0

    pixel_stack = Tuple{Int,Int}[]

    for x in 1:W, y in 1:H
        # @show x,y

        # skip if already assigned to a cluster
        cluster[y,x] != 0 && continue

        # create a new cluster
        curr_cluster += 1
        # println("cluster: $curr_cluster")

        @assert isempty(pixel_stack)
        push!(pixel_stack, (y,x))

        while !isempty(pixel_stack)
            (y,x) = pop!(pixel_stack)

            for (dy,dx) in ((0,-1), (-1,0), (0,1), (1,0), (1,1), (1,-1), (-1,1), (-1,-1))
                (x+dx > 0 && y+dy > 0 && x+dx <= W && y+dy <= H) || continue

                # skip if already assigned to a cluster
                cluster[y+dy,x+dx] != 0 && continue
                
                # if the average difference between channel values
                # is greater than a cutoff, then the colors are different
                # and we can't add this to the cluster
                sum(abs.(frame[:,y+dy,x+dx] .- frame[:,y,x]))/3 < threshold || continue
                # isapprox(frame[:,y+dy,x+dx], frame[:,y,x]) || continue

                # add to cluster
                cluster[y+dy,x+dx] = curr_cluster
                push!(pixel_stack, (y+dy,x+dx))
            end
        end
    end
    cluster
end


function particle_filter(num_particles::Int, observed_images::Array{Float64,4}, num_samples::Int)
    C,H,W,T = size(observed_images)
    
    # construct initial observations
    init_obs = Gen.choicemap((1 => :observed_image, observed_images[:,:,:,1]))
    state = Gen.initialize_particle_filter(model, (H,W,1), init_obs, num_particles)

    # steps
    for t in 2:T
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        obs = Gen.choicemap((t => :observed_image, observed_images[:,:,:,t]))
        Gen.particle_filter_step!(state, (H,W,t), (UnknownChange(),), obs)
    end
    
    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end

# observed_images = crop(load_frames("out/benchmarks/frostbite_1"), top=120, bottom=25, left=20)[:,:,:,1:20]
# traces = particle_filter(100, observed_images, 10)


function do_inference(model, frames, amount_of_computation)
    
    (C,H,W,T) = size(frames)

    # Create a choice map that maps model addresses (:y, i)
    # to observed values ys[i]. We leave :slope and :intercept
    # unconstrained, because we want them to be inferred.
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[t => :observed_image] 
        observations[(:y, i)] = y
    end
    
    # Call importance_resampling to obtain a likely trace consistent
    # with our observations.
    (trace, _) = Gen.importance_resampling(model, (xs,), observations, amount_of_computation);
    return trace
end;




function custom_proposal(current_trace, positions, scores)
    object_id = 1
    # pos = {(object_id => :pos_x)} ~ categorical(positions, scores);
end

function grid_proposal(trace, t, object_id)
    gridding_width = 2
    current_x = trace[t => object_id => :pos_x]
    current_y = trace[t => object_id => :pos_y]
    potential_traces = [
        Gen.update(
            trace,
            Gen.choicemap(
                (object_id => :pos_x) => x,
                (object_id => :pos_y) => y,
            )
        )
        for x in current_x-gridding_width:current_x+gridding_width,
            y in current_y-gridding_width:current_y+gridding_width
    ]
    scores = Gen.get_score.(potential_traces)
    
    
    # ... 


    new_trace = Gen.mh(trace, custom_proposal, scores)


end

function gif_of_trace(trace)
    (H,W,T) = get_args(trace)

    @gif for t in 1:T
        observed = colorview(RGB,trace[t => :observed_image])

        # draw!(c, objs)
        plot(observed, xlims=(0,W), ylims=(0,H), ticks=true)
        annotate!(-60, 20, "Step: $t")
        annotate!(-60, 40, "Objects: $(trace[:N])")
    end
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
function crop(img; top=0, bottom=0, left=0, right=0)
    img[:, top:end-bottom, left:end-right, :]
end



# function contiguous_colors(frame)

# end

# frames = crop(load_frames("out/benchmarks/frostbite_1"), top=120, bottom=25, left=20);

# function do_inference()


function grid(traces; ticks=false, annotate=false)

    (H,W,T) = get_args(traces[1])

    @gif for t in 1:T
        plots = []

        for trace in traces
            observed = colorview(RGB,trace[t => :observed_image])
            object_map = zeros(Int, H, W)

            for i in 1:trace[:N]
                pos = trace[t => i => :pos]
                sprite = trace[i => :shape]
                for I in CartesianIndices(sprite)
                    x, y = Tuple(I)
                    if sprite[I] && 0 < pos.y+y-1 <= H && 0 < pos.x+x-1 <= W
                        object_map[pos.y+y-1, pos.x+x-1] = i
                    end 
                end
            end

            observed = color_labels(object_map)

            push!(plots, plot(observed, xlims=(0,W), ylims=(0,H), ticks=ticks))
            annotate && plot!(title="Step: $t\nObjects: $(trace[:N])")
        end

        plot(plots...)
    end
end

# grid([Gen.simulate(model, (66,141,50)) for _=1:4])

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


# trace,_ = Gen.generate(model, (210, 160, 100));

# trace = Gen.simulate(model, (210, 160, 50));

# gif_of_trace(trace)

# grid([Gen.simulate(model, (66,141,50)) for _=1:4], annotate=true)