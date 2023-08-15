using Revise
using Gen
using GenParticleFilters


function get_mask_diff(mask1, mask2, color1, color2)
    #doesn't use color yet
    # isapprox(color1, color2) && return 1
    !isapprox(color1, color2) && return 1.
    size(mask1) != size(mask2) && return 1.
    # average difference in mask
    sum(i -> abs(mask1[i] - mask2[i]), eachindex(mask1)) / length(mask1)
    #look into the similarity of max overlap thing 
end 
    

"""
groups adjacent same-color pixels of a frame into objects
as a simple first pass of object detection
"""


function test_one_involution(frame)
    (C, H, W) = size(frame)
    (cluster, objs, sprites) = process_first_frame(frame)

    cm = build_init_obs(H,W, sprites, objs, frame)

    tr = generate(model, (H, W, 1), cm)[1]


    for repeat in 1:100#0
        #THING U WANNA TEST HERE 
        #tr, accepted = mh(tr, get_split_merge, (), split_merge_involution)
        #tr = total_update(tr)
        #@show tr
        
        # heatmap = logpdfmap(ImageLikelihood(), tr[:init => :observed_image], render_trace_frame(tr, 0), 0.1)
        # tr, accepted = mh(tr, dd_add_remove_sprite_random, (heatmap,), dd_add_remove_sprite_involution)
        # if accepted
        #     print("added/removed sprite")
        # end
        tr, accepted = mh(tr, get_add_remove_object, (), add_remove_involution)
        if accepted
            print("added/removed sprite")
        end



        # if accepted
        #     print("split/merge")
        # end
        #render trace
        
        if repeat % 10 == 0

            html_body(html_img(draw(H, W, tr[:init => :init_objs], tr[:init => :init_sprites])))
        end 
    end 
render();
end 


function total_update(tr)
    #do one pass of making a heatmap
    
    heatmap = logpdfmap(ImageLikelihood(), tr[:init => :observed_image], render_trace_frame(tr, 0), 0.1)
    

    #sprite proposals 

    #add/remove sprite 
    # orig_M = tr[:init => :num_sprite_types]
    for _ in 1:1
        tr, accepted = mh(tr, add_remove_sprite_random, (), add_remove_sprite_involution)
    end
    # if accepted
    #     @assert orig_M != tr[:init => :num_sprite_types]
    #     if orig_M > tr[:init => :num_sprite_types]
    #         print("added sprite")
    #     else
    #         print("removed sprite")
    #     end
    # end

    # #dd add/remove sprite 
    # #@show tr
    # tr, accepted = mh(tr, dd_add_remove_sprite_random, (heatmap,), dd_add_remove_sprite_involution)
    # if accepted
    #     print("added/removed sprite")
    # end
    

    for i=1:tr[:init => :num_sprite_types] #some objects need more attention. later don't make this just loop through, sample i as well
    

        #recolor involution 
        # tr, accepted = mh(tr, dd_get_random_new_color, (i,), color_involution)
        tr, accepted = mh(tr, select(:init => :init_sprites => i => :color => :r))
        tr, accepted = mh(tr, select(:init => :init_sprites => i => :color => :g))
        tr, accepted = mh(tr, select(:init => :init_sprites => i => :color => :b))

        #resize involution 
        tr, accepted = mh(tr, get_random_size, (i,), size_involution)

     #reshape involution 
        for _ in 1:10 #doing more of these 
            tr, accepted = mh(tr, get_random_hi_wi, (i,), mask_involution)
        end 
    end 

    #object proposals 

    #add/remove object involution 
    # orig_N = tr[:init => :N]
    tr, accepted = mh(tr, get_add_remove_object, (), add_remove_involution)
    # if accepted
    #     @assert orig_N != tr[:init => :N]
    #     if orig_N > tr[:init => :N]
    #         print("added obj")
    #     else
    #         print("removed obj")
    #     end
    # end

    #relayer order objects
    tr, accepted = mh(tr, get_layer_swap, (), layer_involution)

    for i=1:tr[:init => :N]
        #shift objects involution 

        tr, accepted = mh(tr, dd_get_drift, (i, heatmap,), dd_shift_involution) 

        #resprite object involution ok. 
        tr, accepted = mh(tr, select((:init => :init_objs => i => :sprite_index)))
    end 

    tr
end 


function process_first_frame_v2(frame, threshold=.05; num_particles=8, steps=1000, step_chunk=50)
    #run update detect a bunch TODO 
    #@show num_particles, steps, step_chunk
    (C, H, W) = size(frame)

    #something like this but edit
    init_obs = choicemap(
        (:init => :observed_image, frame),
        
        #have perfect build init obs here for everything else and one by one delete 
        #(:init => :N, length(objs)),
        #(:init => :num_sprite_types, length(sprites)),
    )

    # (cluster, objs, sprites) = process_first_frame(frame)
    # init_obs = build_init_obs(H,W, sprites, objs, frame)


    traces = [generate(model, (H, W, 1), init_obs)[1] for _ in 1:num_particles]

    table = fill("", num_particles*2, 1)
    for i in 1:num_particles
        table[i*2,1] = "Particle $i"
    end

    # stats = [Stats() for _ in 1:num_particles]
    # Gen.track_inf = Dict()
    # Gen.track_total = Dict()
    Gen.tracker = Dict()

    elapsed=@elapsed for i in 1:steps
        if i % step_chunk == 1 || i == steps
            println("update: $i")
            col = String[]
            for tr in traces
                push!(col, html_img(draw(H, W, tr[:init => :init_objs], tr[:init => :init_sprites])))
                push!(col, "step=$i<br>N=$(tr[:init => :N])<br>sprites=$(tr[:init => :num_sprite_types])")
            end
            table = hcat(table, col)
        end
        for j in 1:num_particles
            Gen.curr_trace = j
            traces[j] = total_update(traces[j])
            # N = traces[j][:init => :N]
            # M = traces[j][:init => :num_sprite_types]
            # sprite_area = sum([traces[j][:init => :init_sprites => k => :width] * traces[j][:init => :init_sprites => k => :height] for k in 1:M],init=0)
            # @show N
            # @show M
            # @show sprite_area

            # @show length(traces[j][:init => :init_objs])
            # @show length(traces[j][:init => :init_sprites])
        end
    end

    secs_per_step = round(elapsed/steps,sigdigits=3)
    fps = round(1/secs_per_step,sigdigits=3)
    time_str = "MCMC runtime: $(round(elapsed,sigdigits=3))s; $(secs_per_step)s/step; $fps step/s ($num_particles particles))"
    println(time_str)

    #TODO MAKE 2 other tables so this isn't uglyy 

    othertable = fill("", num_particles + 1, 8)

    tracenum = 0

    for tr in traces 
        tracenum += 1 
        # @show tr[:init => :init_objs]
        # @show tr[:init => :init_sprites]
        # @show H, W
        #@show obj_frame(tr[:init => :init_objs], tr[:init => :init_sprites], H, W)
        #@show color_labels(obj_frame(tr[:init => :init_objs], tr[:init => :init_sprites], H, W))
        
        bboxes = html_img(draw_bboxes(render_trace_frame(tr, 0), tr[:init => :init_objs], tr[:init => :init_sprites], 1, H, 1, W))

        objcoloring = html_img(color_labels(obj_frame(tr[:init => :init_objs], tr[:init => :init_sprites], H, W))[1])
        spritecoloring = html_img(color_labels(sprite_frame(tr[:init => :init_objs], tr[:init => :init_sprites], H, W))[1])
        heatmap = render_heatmap(logpdfmap(ImageLikelihood(), tr[:init => :observed_image], render_trace_frame(tr, 0), 0.1))
        diff = html_img(img_diff(tr[:init => :observed_image], render_trace_frame(tr, 0)))
        
        show_stats = ""
        for key in sort(collect(keys(Gen.tracker)))
            key[1] != tracenum && continue
            show_stats != "" && (show_stats *= "<br>")
            stats = Gen.tracker[key]
            show_stats *= "$(key[2]): $(stats.accepted)/$(stats.total) [inf=$(stats.inf); nan=$(stats.nan)]"
        end

        othertable[tracenum + 1, :] = ["Particle $tracenum at step=$steps", html_img(render_trace_frame(tr, 0)), bboxes, objcoloring, spritecoloring, heatmap, diff, show_stats]

        othertable[1, :] = ["", "Rendered", "Bounding Boxes", "Coloring by object", "Coloring by sprite", "logpdf heatmap", "(actual - rendered)", "Stats"]

        #heatmap 

    end 


    # statstable = fill("", num_particles + 1, 7)
    
    html_body(html_table(othertable))
    html_body(html_table(table))
    html_body(time_str)

    # for (k,total) in Gen.track_total
    #     infs = Gen.track_inf[k]
    #     html_body("<br>-inf for $k: $infs/$total")
    # end


end 

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

            for (dy,dx) in ((0,-1), (-1,0), (0,1), (1,0), (1,1), (1,-1), (-1,1), (-1,-1), (0,0))
                (x+dx > 0 && y+dy > 0 && x+dx <= W && y+dy <= H) || continue

                # skip if already assigned to a cluster
                cluster[y+dy,x+dx] != 0 && continue
                
                # if the average difference between channel values
                # is greater than a cutoff, then the colors are different
                # and we can't add this to the cluster

                # #fusing version
                #sum(abs.(frame[:,y+dy,x+dx] .- frame[:,y,x]))/3 < threshold || continue

                #nonfusing version 
                sum(i -> abs(frame[i, y+dy, x+dx] - frame[i, y, x]), eachindex(frame[:, y, x])) / 3 < threshold || continue 

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
    sprites = Sprite[]

    background = 0
    background_size = 0
    for c in 1:curr_cluster
        # create the sprite mask and crop it so its at 0,0
        mask = (cluster .== c)[smallest_y[c]:largest_y[c], smallest_x[c]:largest_x[c]]
        # avg color
        #uncomment. correct code 
        color[c] ./= sum(mask)
        #@show color[c]
        #make array of 3 random numbers to assign to color

        ##random colors for testing 
        #color[c] = rand(Float64, (3)) 

        # largest area sprite is background
        if sum(mask) > background_size
            background = c
            background_size = sum(mask)
        end

        #v2 same sprite type if similar masks 
        newsprite = true
        mask_diff = 0.
        for (i,sprite) in enumerate(sprites)

            #difference in sprite masks
            mask_diff = get_mask_diff(sprite.mask, mask, sprite.color, color[c])
            # @show mask_diff

            #same sprite
            if mask_diff < 0.1
                newsprite = false
                object = Object(i, Position(smallest_y[c], smallest_x[c]))
                push!(objs, object)
                break
            end

            iH,iW = size(sprite.mask)
            cH,cW = size(mask)


        end	


        #new sprite 
        if newsprite
            println("newsprite $(length(sprites)) from cluster $c")
            sprite_type = Sprite(mask, color[c])
            push!(sprites, sprite_type)
            object = Object(length(sprites), Position(smallest_y[c], smallest_x[c]))
            push!(objs, object)
        end

    end

    # turn background into a big rectangle filling whole screen

    #i think works even with spriteindex 
    color = sprites[background].color
    sprites[background] = Sprite(ones(Bool, H, W),color)
    objs[background] = Object(background, Position(1,1))

    (cluster,objs,sprites)	

end

function build_init_obs(H,W, sprites, objs, first_frame)
    init_obs = choicemap(
        (:init => :observed_image, first_frame),
        (:init => :N, length(objs)),
        (:init => :num_sprite_types, length(sprites)),
    )

    for (i,obj) in enumerate(objs)
        # @show obj.sprite_index
        @assert 0 < obj.pos.x <= W && 0 < obj.pos.y <= H
        init_obs[(:init => :init_objs => i => :pos)] = obj.pos
        init_obs[(:init => :init_objs => i => :sprite_index)] = obj.sprite_index #anything not set here it makes a random choice about

        #EDIT THIS 
        init_obs[:init => :init_sprites => obj.sprite_index => :mask] = sprites[obj.sprite_index].mask # => means go into subtrace, here initializing subtraces, () are optional. => means pair!!
        init_obs[:init => :init_sprites => obj.sprite_index => :color] = sprites[obj.sprite_index].color
    end
    init_obs
end

"""
Runs a particle filter on a sequence of frames
"""
function particle_filter(num_particles::Int, observed_images::Array{Float64,4}, num_samples::Int)
    C,H,W,T = size(observed_images)

    # construct initial observations
    (cluster, objs, sprites) = process_first_frame(observed_images[:,:,:,1])

    init_obs = build_init_obs(H,W, sprites, objs, observed_images[:,:,:,1])

    #new version 

    # init_obs = process_first_frame_v2(observed_images[:,:,:,1])


    state = pf_initialize(model, (H,W,1), init_obs, num_particles)

    # steps
    elapsed=@elapsed for t in 1:T-1
        @show t
        # @show state.log_weights, weights
        # maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)
        obs = choicemap((:steps => t => :observed_image, observed_images[:,:,:,t]))
        # particle_filter_step!(state, (H,W,t), (NoChange(),NoChange(),UnknownChange()), obs)
        @time pf_update!(state, (H,W,t+1), (NoChange(),NoChange(),UnknownChange()),
            obs, grid_proposal, (obs,))
    end

    (_, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_normalized_weights)

    # print and render results

    secs_per_step = round(elapsed/(T-1),sigdigits=3)
    fps = round(1/secs_per_step,sigdigits=3)
    time_str = "particle filter runtime: $(round(elapsed,sigdigits=3))s; $(secs_per_step)s/frame; $fps fps ($num_particles particles))"
    println(time_str)
    
    html_body("<p>C: $C, H: $H, W: $W, T: $T</p>")
    html_body("<h2>Observations</h2>", html_gif(observed_images))
    types = map(i -> objs[i].sprite_index, cluster);
    #@show cluster .- types 
    html_body("<h2>First Frame Processing</h2>",
    html_table(["Observation"                       "Objects"                       "Types";
             html_img(observed_images[:,:,:,1])  html_img(color_labels(cluster)[1])  html_img(color_labels(types)[1])
    ]))
    html_body("<h2>Reconstructed Images</h2>")

    table = fill("", 2, length(state.traces))
    for (i,trace) in enumerate(state.traces)
        table[1,i] = "Particle $i ($(round(weights[i],sigdigits=4)))"
        table[2,i] = html_gif(render_trace(trace));
    end

    html_body(html_table(table))
    html_body(time_str)
    
    return sample_unweighted_traces(state, num_samples)
end

import FunctionalCollections

"""
Does gridding to propose new positions for an object in the vicinity
of the current position
"""
@gen function grid_proposal(prev_trace, obs)

    (H,W,prev_T) = get_args(prev_trace)

    t = prev_T
    grid_size = 2
    observed_image = obs[(:steps => t => :observed_image)]

    prev_objs = objs_from_trace(prev_trace, t - 1)
    prev_sprites = sprites_from_trace(prev_trace, 0)

    # now for each object, propose and sample changes 
    for obj_id in 1:prev_trace[:init => :N]

        # get previous position
        if t == 1
            prev_pos = prev_trace[:init => :init_objs => obj_id => :pos]
        else
            prev_pos = prev_trace[:steps => t - 1 => :objs => obj_id => :pos]
        end

        #way to get positions that avoids negatives, todo fix 
        positions = Position[]
        for dx in -grid_size:grid_size,
            dy in -grid_size:grid_size
            #hacky since it cant handle negatives rn
            if prev_pos.x + dx < 1 || prev_pos.x + dx > W || prev_pos.y + dy < 1 || prev_pos.y + dy > H
                push!(positions, Position(prev_pos.y, prev_pos.x))
            else
                push!(positions, Position(prev_pos.y+dy, prev_pos.x+dx))
            end
        end
        
        # flatten
        positions = reshape(positions, :)
        
        # # total update slower version 
        # traces = [Gen.update(trace,choicemap((:steps => t => :objs => obj_id => :pos) => pos))[1] for pos in positions]

        #manually update, partial draw
        scores = Float64[]
        #for each position, score the new image section around the object 
        for pos in positions
            #making the objects with just that object moved 
            objects_one_moved = prev_objs[:]
            objects_one_moved[obj_id] = Object(objects_one_moved[obj_id].sprite_index, pos)

            (sprite_height, sprite_width) = size(prev_sprites[prev_objs[obj_id].sprite_index].mask)
            
            (_, H, W) = size(observed_image)

            #making the little box to render 
            relevant_box_min_y = min(pos.y, prev_pos.y)
            relevant_box_max_y = min(max(pos.y + sprite_height, prev_pos.y + sprite_height), H)
            relevant_box_min_x = min(pos.x, prev_pos.x)
            relevant_box_max_x = min(max(pos.x + sprite_width, prev_pos.x + sprite_width), W)

            drawn_moved_obj = draw_region(objects_one_moved, prev_sprites, relevant_box_min_y, relevant_box_max_y, relevant_box_min_x, relevant_box_max_x) 
            #is it likely
            score = Gen.logpdf(image_likelihood, observed_image[:, relevant_box_min_y:relevant_box_max_y, relevant_box_min_x:relevant_box_max_x], drawn_moved_obj, 0.1)#can I hardcode that

            push!(scores, score)
        end 
        #@show scores 
        #update sprite index? todo?
        
        #making the scores into probabilities 
        scores_logsumexp = logsumexp(scores)
        #@show scores_logsumexp
        scores =  exp.(scores .- scores_logsumexp)
        #these are all zero but one. should it be that way? 
        #@show scores 
        #@show positions

        #oldver
        #scores = Gen.normalize_weights(get_score.(traces))[2]
        #scores = exp.(scores)

        # sample the actual position
        pos = {:steps => t => :objs => obj_id => :pos} ~ labeled_cat(positions, scores)


        # old version: set the curr `trace` to this trace for future iterations of this loop
        # idx = findfirst(x -> x == pos, positions)
        # objs[obj_id] = Object(objs[obj_id].sprite, positions[idx]) 
        # trace = traces[idx]
    end
    nothing 

    #for each sprite type, propose and sample changes
    #todo 
end


# end # module Inference


