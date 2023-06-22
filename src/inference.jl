# frame = crop(load_frames("out/benchmarks/frostbite_1"), top=120, bottom=25, left=20)[:,:,:,20]
# color_labels(process_first_frame(frame),frame)
using Revise
using Gen

# this silliness is necessary for revise(I) to work
try
    includet("model.jl")
catch e
    if isa(e, ErrorException) && occursin("is not a file", e.msg)
        includet("src/model.jl")
    else
        rethrow(e)
    end
end

module I

using ..M
import ..M: Object, Sprite, Position, model
using Gen
import FunctionalCollections: peek

include("html.jl")

# @gen function baz(x)
#     x = {:x} ~ normal(30, 1)
#     x
# end

"""
groups adjacent same-color pixels of a frame into objects
as a simple first pass of object detection
"""
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

    # (start_y, start_x, end_y, end_x)
    smallest_y = [typemax(Int) for _ in 1:curr_cluster]
    smallest_x = [typemax(Int) for _ in 1:curr_cluster]
    largest_y = [typemin(Int) for _ in 1:curr_cluster]
    largest_x = [typemin(Int) for _ in 1:curr_cluster]
    color = [[0.,0.,0.] for _ in 1:curr_cluster]

    for x in 1:W, y in 1:H
        c = cluster[y,x]
        smallest_y[c] = min(smallest_y[c], y)
        smallest_x[c] = min(smallest_x[c], x)
        largest_y[c] = max(largest_y[c], y)
        largest_x[c] = max(largest_x[c], x)
        color[c] += frame[:,y,x]
    end

    objs = Object[]

    background = 0
    background_size = 0
    for c in 1:curr_cluster
        # create the sprite mask and crop it so its at 0,0
        mask = (cluster .== c)[smallest_y[c]:largest_y[c], smallest_x[c]:largest_x[c]]
        # avg color
        color[c] ./= sum(mask)
        # largest area sprite is background
        if sum(mask) > background_size
            background = c
            background_size = sum(mask)
        end
        sprite = Sprite(mask, color[c])
        object = Object(sprite, Position(smallest_y[c], smallest_x[c]))
        push!(objs, object)
    end

    # turn background into a big rectangle filling whole screen
    color = objs[background].sprite.color
    objs[background] = Object(Sprite(ones(Bool, H, W),color), Position(1,1))

    (cluster,objs)
end

"""
Runs a particle filter on a sequence of frames
"""
function particle_filter(num_particles::Int, observed_images::Array{Float64,4}, num_samples::Int)
    C,H,W,T = size(observed_images)

    html = new_html()
    
    # construct initial observations
    (cluster, objs) = process_first_frame(observed_images[:,:,:,1])
    init_obs = choicemap(
        (:init => :observed_image, observed_images[:,:,:,1]),
        (:N, length(objs)),
        (:width => W),
        (:height => H),
    )
    # @show W,H
    for (i,obj) in enumerate(objs)
        @assert 0 < obj.pos.x <= W && 0 < obj.pos.y <= H
        # @show i,obj.pos
        init_obs[(:init => :objs => i => :pos)] = obj.pos
        init_obs[(i => :shape)] = obj.sprite.mask
        init_obs[(i => :color)] = obj.sprite.color
    end

    first_frame = html_img(html, observed_images[:,:,:,1])
    add_body!(html, "<h2>Observations</h2>", html_gif(html, observed_images))
    

    # printstyled("initializing particle filter\n",color=:green, bold=true)
    state = initialize_particle_filter(model, (H,W,1), init_obs, num_particles)

    # steps
    for t in 1:T-1
        @show t
        # @show state.log_weights, weights
        # maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)
        obs = choicemap((:steps => t => :observed_image, observed_images[:,:,:,t]))
        # particle_filter_step!(state, (H,W,t), (NoChange(),NoChange(),UnknownChange()), obs)
        @time particle_filter_step!(state, (H,W,t+1), (NoChange(),NoChange(),UnknownChange()),
            obs, grid_proposal, (obs,))
    end

    (_, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_normalized_weights)


    table = fill("", 2, length(state.traces))
    for (i,trace) in enumerate(state.traces)
        table[1,i] = "Particle $i ($(round(weights[i],sigdigits=4)))"
        table[2,i] = html_gif(html, M.render_trace(trace));
    end

    add_body!(html, html_table(html, table))

    render(html)
    
    # return rand(state.traces, num_samples)
    
    return sample_unweighted_traces(state, num_samples)
end

# observed_images = crop(load_frames("out/benchmarks/frostbite_1"), top=120, bottom=25, left=20)[:,:,:,1:20]
# traces = particle_filter(100, observed_images, 10)



"""
Does gridding to propose new positions for an object in the vicinity
of the current position
"""
@gen function grid_proposal(prev_trace, obs)

    (H,W,prev_T) = get_args(prev_trace)
    # @show prev_T

    if prev_T == 1
        objs = M.objs_from_trace(prev_trace, 0)
    else
        objs = peek(get_retval(prev_trace)).objs[:]
    end
    

    t = prev_T
    grid_size = 2
    observed_image = obs[(:steps => t => :observed_image)]

    # first get a proposal from the prior by just extending the trace by one timestep and also adding the new observation in
    (trace, _, _, _) = Gen.update(prev_trace, (H,W,prev_T + 1), (NoChange(), NoChange(), UnknownChange()), obs)

    # display(grid([trace]))

    # now for each object, propose and sample changes 
    for obj_id in 1:trace[:N]
        # we use the prev_trace position here actually!
        if t == 1
            prev_pos = prev_trace[:init => :objs => obj_id => :pos]
        else
            prev_pos = prev_trace[:steps => t - 1 => :objs => obj_id => :pos]
        end
        positions = [
            Position(prev_pos.y+dy, prev_pos.x+dx)
            for dx in -grid_size:grid_size,
                dy in -grid_size:grid_size
        ]
        # flatten
        positions = reshape(positions, :)
        
        # base_score = sum(abs.(Array(channelview(M.draw!(M.canvas(H, W), objs))) - observed_image))

        # scores = Float64[]
        # for pos in positions
        #     objs[obj_id] = Object(objs[obj_id].sprite, pos)
        #     rendered = Array(channelview(M.draw!(M.canvas(H, W), objs)))
        #     score = exp(sum(abs.(rendered - observed_image)) - base_score)
        #     push!(scores, score)
        # end

        # scores ./= sum(scores)
        # @show scores

        # @show obj_id
        traces = [Gen.update(trace,choicemap((:steps => t => :objs => obj_id => :pos) => pos))[1] for pos in positions]
        scores = Gen.normalize_weights(get_score.(traces))[2]
        scores = exp.(scores)

        # sample the actual position
        pos = {:steps => t => :objs => obj_id => :pos} ~ M.labeled_cat(positions, scores)

        # set the curr `trace` to this trace for future iterations of this loop
        idx = findfirst(x -> x == pos, positions)
        # objs[obj_id] = Object(objs[obj_id].sprite, positions[idx]) 
        trace = traces[idx]
    end
    nothing 
end


end # module Inference