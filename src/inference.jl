using Revise
using Gen
using GenParticleFilters
using GenSMCP3

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


        
        # #v0 old version 
        # sprite = Sprite(mask, color[c])
        # object = Object(sprite, Vec(smallest_y[c], smallest_x[c]))
        # push!(objs, object)


        # #v1 each sprite makes new sprite type version 
        # sprite = Sprite(mask, color[c])
        # object = Object(c, Vec(smallest_y[c], smallest_x[c]))
        # push!(sprites, sprite)
        # push!(objs, object)


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
                object = Object(i, Vec(smallest_y[c], smallest_x[c]))
                push!(objs, object)
                break
            end

            iH,iW = size(sprite.mask)
            cH,cW = size(mask)

            #checking for subsprites in either direction for occlusion # not fully working, feel free to delete since takes long 
            check_for_subsprites = false 
            if check_for_subsprites
                if iH < cH || iW < cW
                    smallmask = sprite.mask
                    bigmask = mask
                    smallH = iH
                    bigH = cH
                    smallW = iW
                    bigW = cW
                    newbigger = true
                else 
                    smallmask = mask
                    bigmask = sprite.mask
                    smallH = cH
                    bigH = iH
                    smallW = cW
                    bigW = iW
                    newbigger = false
                end

                for Hindex in 1:bigH-smallH+1
                    for Windex in 1:bigW-smallW+1
                        submask = bigmask[Hindex:Hindex+smallH-1, Windex:Windex+smallW-1] #check indicies here 
                        mask_diff = get_mask_diff(submask, smallmask, sprite.color, color[c])
                        # @show mask_diff	

                        if mask_diff < 0.2
                            println("holy shit this actually worked")
                            newsprite = false

                            if newbigger
                                #fixing old sprite type #todo should also fix its pos 
                                sprites[i] = Sprite(bigmask,sprite.color)
                                object = Object(i, Vec(smallest_y[c], smallest_x[c]))
                                push!(objs, object)
                            else 
                                #new sprite is old just starting at diff index
                                object = Object(i, Vec(max(smallest_y[c]-Hindex, 1), max(smallest_x[c]-Windex, 1))) #looks weird when <0
                                
                                push!(objs, object)
                            end 
                            break 
                            
                        end
                    end 
                end
            end


        end	


        #new sprite 
        if newsprite
            println("newsprite $(length(sprites)) from cluster $c")
            sprite = Sprite(mask, color[c])
            push!(sprites, sprite)
            object = Object(length(sprites), Vec(smallest_y[c], smallest_x[c]))
            push!(objs, object)
        end

    end

    # turn background into a big rectangle filling whole screen

    #i think works even with spriteindex 
    color = sprites[background].color
    sprites[background] = Sprite(ones(Bool, H, W),color)
    objs[background] = Object(background, Vec(1,1))

    (cluster,objs,sprites)	

end

function build_init_obs(H,W, sprites, objs, observed_images)
    init_obs = choicemap(
        (:init => :observed_image, observed_images[:,:,:,1]),
        (:init => :N, length(objs)),
        (:init => :num_sprites, length(sprites)),
    )

    for (i,obj) in enumerate(objs)
        # @show obj.sprite_index
        @assert 0 < obj.pos.x <= W && 0 < obj.pos.y <= H
        init_obs[(:init => :init_objs => i => :pos)] = obj.pos
        init_obs[(:init => :init_objs => i => :sprite_index)] = obj.sprite_index #anything not set here it makes a random choice about

        #EDIT THIS 
        init_obs[:init => :init_sprites => obj.sprite_index => :shape] = sprites[obj.sprite_index].mask # => means go into subtrace, here initializing subtraces, () are optional. => means pair!!
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

    init_obs = build_init_obs(H,W, sprites, objs, observed_images)

    state = pf_initialize(model, (H,W,1), init_obs, num_particles)

    # @show get_choices(state.traces[1])

    # steps
    elapsed=@elapsed for t in 1:T-1
        @show t
        # @show state.log_weights, weights
        # maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)
        obs = choicemap((:steps => t => :observed_image, observed_images[:,:,:,t]))
        # @time pf_update!(state, (H,W,t+1), (NoChange(),NoChange(),UnknownChange()),
        #     obs, fwd_proposal, (obs,))

        # Rejuvenation
        # todo currently not getting "credit" for these rejuv steps
        for i in 1:num_particles
            rejuv(state.traces[i])
        end

        

        @time pf_update!(state, (H,W,t+1), (NoChange(),NoChange(),UnknownChange()),
            obs, SMCP3Update(
                fwd_proposal_naive,
                bwd_proposal_naive,
                (obs,),
                (obs,),
                false, # check are inverses
            ))
        
        
        #render 
        #html_body(html_img())
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

function rejuv(trace)
    return trace, 0.
end

@inline function get_pos(trace, obj_id, t)
    if t == 0
        trace[:init => :init_objs => obj_id => :pos]
    else
        trace[:steps => t => :objs => obj_id => :pos]
    end
end

const SAMPLES_PER_OBJ = 40


@kernel function fwd_proposal_naive(prev_trace, obs)
    # @show "fwd proposal timee"
    # return (
    #     choicemap((:init => :N, prev_trace[:init => :N]+1)),
    #     choicemap()
    # )

    (H,W,prev_T) = get_args(prev_trace)
    t = prev_T
    observed_image = obs[(:steps => t => :observed_image)]

    # prev_objs = state_of_trace(prev_trace, t - 1).objs
    # prev_state = state_of_trace(prev_trace, t - 1)
    # prev_sprites = env_of_trace(prev_trace).sprites
    curr_env = deepcopy(env_of_trace(prev_trace))

    bwd_choices = choicemap()
    trace_updates = choicemap()


    # @show typeof(prev_trace.trace)

    for obj_id in eachindex(curr_env.state.objs)

        curr_state = curr_env.state
        states = []
        constraints = []
        steps = []
        for j in 1:SAMPLES_PER_OBJ

            # env.state = deepcopy(prev_state)
            # for i in eachindex(env.state.objs)
            #     {:objs => i} ~ obj_dynamics(i, env)
            # end
            # for i in eachindex(env.state.objs)
            #     pos = {:pos_noise => i} ~ normal_vec(env.state.objs[i].pos, 1.0)
            #     #sprite noise? 
            #     env.state.objs[i].pos = pos
            # end
        
            # rendered = draw(canvas_height, canvas_width, env.state.objs, env.sprites)
            # observed_image ~ image_likelihood(rendered, var)

            curr_env.state = deepcopy(curr_state) # can be less of a deepcopy
            curr_env.step_of_obj[obj_id] = {(:func, obj_id, j)} ~ uniform_discrete(1, length(curr_env.code_library))
            {:dynamics => obj_id => j} ~ obj_dynamics(obj_id, curr_env) #wanna keep these trace choices around
             
            
            #trace_updates[:steps => t+1 => :random_stuff => obj_id => j => :dynamics] #UPDATE THIS WITH THE ABOVE??
            
            # todo maybe add the noise

            push!(states, curr_env.state) #env.state changed
            push!(constraints, curr_env.constraints)
            push!(steps, curr_env.step_of_obj[obj_id])
        end

        html_body("<br>moving obj $obj_id<br>")
        scores = Float64[]
        for state in states
            rendered = draw(H, W, state.objs, curr_env.sprites) #eventually only draw part 
            html_body(html_img(rendered))
            score = Gen.logpdf(image_likelihood, observed_image, rendered, 0.1) 
            html_body("$score")
            push!(scores, score)
        end

        #sample from scores 
        scores_logsumexp = logsumexp(scores)
        scores =  exp.(scores .- scores_logsumexp)
        @show scores 

        idx = {:idx => obj_id} ~ categorical(scores)
        curr_env.state = states[idx] #give this state onwards in the loop for the next obj, doesnt rly matter
        # curr_env.constraints = constraints[idx]
        # curr_env.step_of_obj[obj_id] = steps[idx]

        # @show trace_updates
        # @show prev_trace
        # trace_updates[:steps] = "test"
        

        # @show t obj_id
        # trace_updates[:steps => t => :objs => obj_id] = constraints[idx]
        set_submap!(trace_updates,:steps => t => :objs => obj_id, constraints[idx])
        trace_updates[:init => :init_objs => obj_id => :step_of_obj] = steps[idx]
        # @show constraints[idx]


        # set_(trace_updates,:steps => t => :objs => obj_id, constraints[idx])
        #trace_updates[:steps => observed_image] = "test"
        
        

        # trace_updates[:steps => t+1 => :random_stuff => (:idx_chosen_for_obj, obj_id)] = idx
        # obj = curr_env.state.objs[obj_id]
        # trace_updates[:steps => t+1 => :objs => obj_id => :pos] = obj.pos
        # trace_updates[:steps => t+1 => :objs => obj_id => :sprite_index] = obj.sprite_index
        
        #trace_updates[:steps => t+1 => :objs => obj_id => :attrs] = uh

        
        
    end


    html_body("<br>Result<br>")
    html_body(html_img(draw(H, W, curr_env.state.objs, curr_env.sprites)))
    # Gen.update(prev_trace, trace_updates, ())

    # Gen.set_retval!()

    # prev_trace.trace[:init] = curr_env
    # prev_trace.trace[:steps => t] = curr_env.state

    #trace_updates[:steps => t+1 => :objs] = curr_state.objs 
    

    #bwd_choices delete last step 

    # Gen.update(prev_trace, trace_updates)

    return (
        trace_updates, # choicemap
        bwd_choices
    )
end
        
        

    





        # # manually update, partial draw
        # scores = Float64[]
        # #for each position, score the new image section around the object 
        # (sprite_height, sprite_width) = size(prev_sprites[prev_objs[obj_id].sprite_index].mask)

        # # get a shared bounding box around all the positions
        # ((box_min_y, box_min_x),(box_max_y, box_max_x)) = pixel_vec.(inbounds_vec.(min_max_vecs(positions), H, W))
        # box_max_y = min(box_max_y + sprite_height, H)
        # box_max_x = min(box_max_x + sprite_width, W)

        # objects_one_moved = prev_objs[:]
        # cropped_obs = observed_image[:, box_min_y:box_max_y, box_min_x:box_max_x]

        # for pos in positions
        #     # move the object
        #     objects_one_moved[obj_id] = set_pos(objects_one_moved[obj_id], pos)
        #     drawn_moved_obj = draw_region(objects_one_moved, prev_sprites, box_min_y, box_max_y, box_min_x, box_max_x) 
        #     # todo var=0.1 is hardcoded for now
        #     score = Gen.logpdf(image_likelihood, cropped_obs, drawn_moved_obj, 0.1) 
        #     push!(scores, score)
        # end 
        
        # #making the scores into probabilities 
        # scores_logsumexp = logsumexp(scores)
        # scores =  exp.(scores .- scores_logsumexp)

        # # sample the actual position
        # idx = {obj_id => :idx} ~ categorical(scores)
        # bwd_choices[obj_id => :idx] = idx

        # trace_updates[:steps => t => :objs => obj_id => :pos] = positions[idx]
    
    
    # end
    

    # return (
        
    #     trace_updates, # (:init => :N, prev_trace[:init => :N]+1) # a choice map 
    #     # bwd proposal doesnt need to be passed any extra information to invert the update! No matter
    #     # what choices it makes itll be able to invert the update, since itll just be shortening the
    #     # trace by one step as opposed to doing something like probabilistically adjusting :N etc.
    #     bwd_choices
    # )
# end

@kernel function bwd_proposal_naive(next_trace, obs)
    return (
        choicemap(),
        choicemap()
    );

    (H,W,next_T) = get_args(next_trace)
    t = next_T - 1 

    fwd_choices = choicemap()

    for obj_id in 1:next_trace[:init => :N]

        prev_pos = get_pos(next_trace, obj_id, t-1)

        # sample a random index - uniform bc true posterior is uniform
        # add to choicemap: the index; and putting the fwd trace value at that index
        idx = {obj_id => :idx} ~ uniform_discrete(1, SAMPLES_PER_OBJ)
        fwd_choices[obj_id => :idx] = idx

        for j in 1:SAMPLES_PER_OBJ
            if j == idx
                fwd_choices[obj_id => j] = next_trace[:steps => t => :objs => obj_id => :pos]
            else
                # potential problem: if one of those 19 were way better than the chosen one, itll seem like 
                # the fwd proposal did something super unlikely
                fwd_choices[obj_id => j] = {obj_id => j} ~  normal_vec(prev_pos, .5)
            end
        end

    end


    return (
        # no need to pass any extra info to invert the update! The update would be inverted simply
        # by the change in arguments (when it gets prev_t instead of t). Since things like :N or
        # other variables aren't changed, and this is simply extending the existing trace, we don't
        # need anything here!
        choicemap(),
        # 
        fwd_choices
    )
end

# @kernel function fwd_proposal(prev_trace, obs)

#     (H,W,prev_T) = get_args(prev_trace)
#     t = prev_T
#     grid_size = 2
#     observed_image = obs[(:steps => t => :observed_image)]

#     prev_objs = objs_from_trace(prev_trace, t - 1)
#     prev_sprites = sprites_from_trace(prev_trace, 0) # todo t=0 for now bc no changing sprites over time

#     # now for each object, propose and sample changes 
#     for obj_id in 1:prev_trace[:init => :N]

#         # get previous position
#         if t == 1
#             prev_pos = prev_trace[:init => :init_objs => obj_id => :pos]
#         else
#             prev_pos = prev_trace[:steps => t - 1 => :objs => obj_id => :pos]
#         end

#         #way to get positions that avoids negatives, todo fix 
#         positions = [Vec(prev_pos.y+dy, prev_pos.x+dx) for dx in -grid_size:grid_size, dy in -grid_size:grid_size]
        
#         # flatten
#         positions = reshape(positions, :)

#         #manually update, partial draw
#         scores = Float64[]
#         #for each position, score the new image section around the object 
#         (sprite_height, sprite_width) = size(prev_sprites[prev_objs[obj_id].sprite_index].mask)

#         # get a shared bounding box around all the positions
#         ((box_min_y, box_min_x),
#         (box_max_y, box_max_x)) = pixel_vec.(inbounds_vec.(min_max_vecs(positions), H, W))
#         box_max_y = min(box_max_y + sprite_height, H)
#         box_max_x = min(box_max_x + sprite_width, W)

#         objects_one_moved = prev_objs[:]
#         cropped_obs = observed_image[:, box_min_y:box_max_y, box_min_x:box_max_x]

#         for pos in positions
#             # move the object
#             objects_one_moved[obj_id] = set_pos(objects_one_moved[obj_id], pos)
#             drawn_moved_obj = draw_region(objects_one_moved, prev_sprites, box_min_y, box_max_y, box_min_x, box_max_x) 
#             # todo var=0.1 is hardcoded for now
#             score = Gen.logpdf(image_likelihood, cropped_obs, drawn_moved_obj, 0.1) 
#             push!(scores, score)
#         end 
        
#         #making the scores into probabilities 
#         scores_logsumexp = logsumexp(scores)
#         scores =  exp.(scores .- scores_logsumexp)

#         # sample the actual position
#         pos = {:steps => t => :objs => obj_id => :pos} ~ labeled_cat(positions, scores)
#     end
#     nothing
#     # return (
#     #     choicemap(),
#     #     choicemap()
#     # )
# end


"""
Does gridding to propose new positions for an object in the vicinity
of the current position
"""
@gen function grid_proposal(prev_trace, obs)

    (H,W,prev_T) = get_args(prev_trace)

    t = prev_T
    grid_size = 2
    observed_image = obs[(:steps => t => :observed_image)]

    prev_objs = state_of_trace(prev_trace, t - 1)
    prev_sprites = env_of_trace(prev_trace).sprites

    # now for each object, propose and sample changes 
    for obj_id in 1:prev_trace[:init => :N]

        # get previous position
        if t == 1
            prev_pos = prev_trace[:init => :init_objs => obj_id => :pos]
        else
            prev_pos = prev_trace[:steps => t - 1 => :objs => obj_id => :pos]
        end

        #way to get positions that avoids negatives, todo fix 
        positions = Vec[]
        for dx in -grid_size:grid_size,
            dy in -grid_size:grid_size
                push!(positions, Vec(prev_pos.y+dy, prev_pos.x+dx))
        end
        
        # flatten
        positions = reshape(positions, :)
        
        # # total update slower version 
        # traces = [Gen.update(trace,choicemap((:steps => t => :objs => obj_id => :pos) => pos))[1] for pos in positions]

        #manually update, partial draw
        scores = Float64[]
        #for each position, score the new image section around the object 
        (sprite_height, sprite_width) = size(prev_sprites[prev_objs[obj_id].sprite_index].mask)

        # get a shared bounding box around all the positions
        ((box_min_y, box_min_x),
        (box_max_y, box_max_x)) = pixel_vec.(inbounds_vec.(min_max_vecs(positions), H, W))
        box_max_y = min(box_max_y + sprite_height, H)
        box_max_x = min(box_max_x + sprite_width, W)

        objects_one_moved = prev_objs[:]
        cropped_obs = observed_image[:, box_min_y:box_max_y, box_min_x:box_max_x]

        for pos in positions
            # move the object
            objects_one_moved[obj_id] = set_pos(objects_one_moved[obj_id], pos)
            drawn_moved_obj = draw_region(objects_one_moved, prev_sprites, box_min_y, box_max_y, box_min_x, box_max_x) 
            # todo var=0.1 is hardcoded for now
            score = Gen.logpdf(image_likelihood, cropped_obs, drawn_moved_obj, 0.1) 
            push!(scores, score)
        end 

        #update sprite index? todo?
        
        #making the scores into probabilities 
        scores_logsumexp = logsumexp(scores)
        scores =  exp.(scores .- scores_logsumexp)

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


