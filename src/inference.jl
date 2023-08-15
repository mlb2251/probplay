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
        (:init => :init_state => :N, length(objs)),
        (:init => :num_sprites, length(sprites)),
    )

    for (i,obj) in enumerate(objs)
        # @show obj.sprite_index
        @assert 0 < obj.pos.x <= W && 0 < obj.pos.y <= H
        init_obs[(:init => :init_state => :init_objs => i => :pos)] = obj.pos
        init_obs[(:init => :init_state => :init_objs => i => :sprite_index)] = obj.sprite_index #anything not set here it makes a random choice about

        #EDIT THIS 
        init_obs[:init => :init_sprites => obj.sprite_index => :shape] = sprites[obj.sprite_index].mask # => means go into subtrace, here initializing subtraces, () are optional. => means pair!!
        init_obs[:init => :init_sprites => obj.sprite_index => :color] = sprites[obj.sprite_index].color
    end
    init_obs
end

#attributes shift a little instead of full mh resample. not data driven
@gen function variable_shift_randomness(t, addr)
    shift ~ normal(0, 0.1)#play with this number, relate it to v?
end 
function variable_shift_involution(t, forward_choices, forward_retval, proposal_args)
    new_trace_choices = choicemap()
    backward_choices = choicemap()
    addr = proposal_args[1]
    # addr = (:init => :init_state => :init_objs => 1 => :attrs => 1 => :attr)
    # @show addr
    new_trace_choices[addr] = t[addr] + forward_choices[:shift]
    backward_choices[:shift] = -forward_choices[:shift]

    new_trace, weight, = update(t, get_args(t), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
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
        obs = choicemap((:steps => t => :observed_image, observed_images[:,:,:,t+1]))
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

        
        for i in 1:num_particles
            tr = state.traces[i]
            for obj_id in 1:length(objs)
                
                #@show tr
               # @show tr[:init => :init_state => :init_objs => obj_id => :vel]
                if obj_id == 3
                    @show tr[:init => :init_state => :init_objs => obj_id => :attrs]
                end 
                tr, accept = mh(tr, select(:init => :init_state => :init_objs => obj_id => :step_of_obj))
                

                #doing a shifting involution on each attribute 
                for attr_id in 1:1 #just velocity for now 
                    attr_address = (:init => :init_state => :init_objs => obj_id => :attrs => attr_id => :attr)
                    for _ in 1:10
                        tr, accept = mh(tr, variable_shift_randomness, (attr_address,), variable_shift_involution)
                    end 
                end
            end 
            state.traces[i] = tr
            @show tr
        end 

       

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

    html_body("<script>tMax = $T;</script>")
    html_body("<p>C: $C, H: $H, W: $W, T: $T</p>")
    html_body("<h2>Observations</h2>", html_gif(observed_images))




    types = map(i -> objs[i].sprite_index, cluster);
    html_body("<h2>First Frame Processing</h2>",
    html_table(["Observation"                       "Objects"                       "Types";
             html_img(observed_images[:,:,:,1])  html_img(color_labels(cluster)[1])  html_img(color_labels(types)[1])
    ]))
    html_body("<h2>Reconstructed Images</h2>")

    table = fill("", 3, length(state.traces))
    for (i,trace) in enumerate(state.traces)
        table[1,i] = "Particle $i ($(round(weights[i],sigdigits=4)))"
        rendered = render_trace(trace)
        table[2,i] = html_gif(rendered);
        table[3,i] = html_gif(img_diff(rendered, observed_images));
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
        trace[:init => :init_state => :init_objs => obj_id => :pos]
    else
        trace[:steps => t => :objs => obj_id => :pos]
    end
end

const SAMPLES_PER_OBJ = 40 # 1
show_forward_proposals :: Bool = false

#next abstract this
#@gen function sample_from_scores(scores, list, )
#     scores_logsumexp = logsumexp(scores)
#     scores =  exp.(scores .- scores_logsumexp)
#     idx = categorical(scores)
#     return idx
# end
# #sample from scores 
# scores_logsumexp = logsumexp(scores)
# scores =  exp.(scores .- scores_logsumexp)

# idx = {:idx => obj_id} ~ categorical(scores)

# bwd_choices[:idx => obj_id] = idx
# curr_env.state = states[idx] #give this state onwards in the loop for the next obj, doesnt rly matter
# curr_env.step_of_obj = step_of_objs[idx]

@kernel function fwd_proposal_naive(prev_trace, obs)
    
    (H,W,prev_T) = get_args(prev_trace)
    t = prev_T # `t` will be our newly added timestep
    observed_image = obs[(:steps => t => :observed_image)]
    curr_env = deepcopy(env_of_trace(prev_trace))

    """
    t=0 is the first observation so t=1 is the second observation (after the first step)
    `T` is the total number of *observations* (`T-1` is the number of steps)
    """

    bwd_choices = choicemap()
    trace_updates = choicemap()

    if show_forward_proposals

        html_body("<div class=\"proposal\" t=$t>")
        html_body("<h3>Proposal @ t=$t</h3>")
        html_body("Observation (t=$t): <br>", html_img(observed_image), "<br>")
        prev_img = render_trace_frame(prev_trace, t-1) #draw(H, W, env_of_trace(prev_trace).state.objs, curr_env.sprites)
        html_body("Prev State (t=$(t-1)): <br>", html_img(prev_img), "<br>")
        html_body("(Obs - Prev):<br>", html_img(img_diff(observed_image,prev_img)), "<br><br>")
        html_body("Prev Choices:<br>")
        html_body(replace(string(get_choices(prev_trace.trace)),"\n"=>"<br>", "\t" => "&emsp;", " " => "&nbsp;"))

    end


    for obj_id in eachindex(curr_env.state.objs)

        curr_state = curr_env.state
        states = []
        constraints = []
        for j in 1:SAMPLES_PER_OBJ
            curr_env.state = deepcopy(curr_state) # can be less of a deepcopy
         

            {:dynamics => obj_id => j} ~ obj_dynamics(obj_id, curr_env, choicemap())

            {:dynamics => obj_id => j} ~ obj_dynamics(obj_id, curr_env, choicemap())
            set_submap!(bwd_choices, :dynamics => obj_id => j, curr_env.exec.constraints)


            push!(states, curr_env.state) #env.state changed
            push!(constraints, curr_env.exec.constraints)
        end

        scores = Float64[]
        canvas = zeros(Float64, 3, H, W)
        for (i,state) in enumerate(states)
            fill!(canvas, 0.)
            rendered = draw(canvas, state.objs, curr_env.sprites) #eventually only draw part 
            score = Gen.logpdf(image_likelihood, observed_image, rendered, 0.1) 
            push!(scores, score)
        end


        #sample from scores 
        scores_logsumexp = logsumexp(scores)
        scores =  exp.(scores .- scores_logsumexp)

        idx = {:idx => obj_id} ~ categorical(scores)
        
        bwd_choices[:idx => obj_id] = idx
        curr_env.state = states[idx] #give this state onwards in the loop for the next obj, doesnt rly matter
        #curr_env.step_of_obj = step_of_objs[idx]

        
        

        if show_forward_proposals            

            html_body("<br>Proposals for object $obj_id<br>")

            table = fill("", 4, SAMPLES_PER_OBJ)

            for (i,state) in enumerate(states)
                rendered = draw(H, W, state.objs, curr_env.sprites) 
                score = scores[i] #Gen.logpdf(image_likelihood, observed_image, rendered, 0.1) 
                table[1,i] = html_img(rendered)
                table[2,i] = html_img(img_diff(observed_image, rendered))
                table[3,i] = "$(img_diff_sum(observed_image, rendered))"
                if i == idx
                    table[4,i] = "<b>$i: $score</b>"
                else 
                    table[4,i] = "$i: $score"
                end
            end

            html_body(html_table(table))
            html_body("Chose: $idx<br>")

        end

        set_submap!(trace_updates,:steps => t => :objs => obj_id => :step, constraints[idx])
        
    end

    @show [curr_env.step_of_obj[i] for i in eachindex(curr_env.state.objs)]

    if show_forward_proposals
        res = draw(H, W, curr_env.state.objs, curr_env.sprites)
        html_body("<br>Result<br>", html_img(res), "<br><br>")
        # html_body("Heatmap<br>", render_heatmap(logpdfmap(ImageLikelihood(), observed_image, res, .1)), "<br><br>")
        html_body("observed-res<br>", html_img(img_diff(observed_image,res)), "<br><br>")
        html_body(replace(string(trace_updates),"\n"=>"<br>", "\t" => "&emsp;", " " => "&nbsp;"))
        html_body("</div>")
    end

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
        
    #     trace_updates, # (:init => :init_state => :N, prev_trace[:init => :init_state => :N]+1) # a choice map 
    #     # bwd proposal doesnt need to be passed any extra information to invert the update! No matter
    #     # what choices it makes itll be able to invert the update, since itll just be shortening the
    #     # trace by one step as opposed to doing something like probabilistically adjusting :N etc.
    #     bwd_choices
    # )
# end

@kernel function bwd_proposal_naive(next_trace, obs)

    (H,W,next_T) = get_args(next_trace)
    t = next_T - 1 # `t` for the fwd proposal is equal to prev_T, which is next_T-1

    curr_env = env_of_trace(next_trace)
    prev_state = state_of_trace(next_trace, t-1) # get the state *before* the proposal even happened (the state we'll be reverted to after this finishes)

    fwd_choices = choicemap()

    # roll back to prev state
    curr_env.state = prev_state

    for obj_id in eachindex(curr_env.state.objs)

        curr_state = curr_env.state

        # sample a random index where we'll keep the actual subtrace
        idx = {:idx => obj_id} ~ uniform_discrete(1, SAMPLES_PER_OBJ)
        fwd_choices[:idx => obj_id] = idx
        actual_choices = get_submap(get_choices(next_trace), :steps => t => :objs => obj_id => :step)

        for j in 1:SAMPLES_PER_OBJ
            curr_env.state = deepcopy(curr_state) # possibly slightly overkill to do this in the backward pass
            if j == idx
                set_submap!(fwd_choices, :dynamics => obj_id => j, actual_choices)
            else
                # potential problem: if one of those 19 were way better than the chosen one, itll seem like 
                # the fwd proposal did something super unlikely
                {:dynamics => obj_id => j} ~  obj_dynamics(obj_id, curr_env, choicemap())
                set_submap!(fwd_choices, :dynamics => obj_id => j, curr_env.exec.constraints)
            end
        end

        # now run the actual dynamics (untraced, but will make no unconstrained choices)
        # for this object to get the right intermediate state
        curr_env.state = deepcopy(curr_state)
        obj_dynamics(obj_id, curr_env, actual_choices)

    end


    return (
        # no need to pass any extra info to invert the update! The update would be inverted simply
        # by the change in arguments (when it gets prev_t instead of t). Since things like :N or
        # other variables aren't changed, and this is simply extending the existing trace, we don't
        # need anything here!
        choicemap(),
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
#     for obj_id in 1:prev_trace[:init => :init_state => :N]

#         # get previous position
#         if t == 1
#             prev_pos = prev_trace[:init => :init_state => :init_objs => obj_id => :pos]
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
# @gen function grid_proposal(prev_trace, obs)

#     (H,W,prev_T) = get_args(prev_trace)

#     t = prev_T
#     grid_size = 2
#     observed_image = obs[(:steps => t => :observed_image)]

#     prev_objs = state_of_trace(prev_trace, t - 1)
#     prev_sprites = env_of_trace(prev_trace).sprites

#     # now for each object, propose and sample changes 
#     for obj_id in 1:prev_trace[:init => :init_state => :N]

#         # get previous position
#         if t == 1
#             prev_pos = prev_trace[:init => :init_state => :init_objs => obj_id => :pos]
#         else
#             prev_pos = prev_trace[:steps => t - 1 => :objs => obj_id => :pos]
#         end

#         #way to get positions that avoids negatives, todo fix 
#         positions = Vec[]
#         for dx in -grid_size:grid_size,
#             dy in -grid_size:grid_size
#                 push!(positions, Vec(prev_pos.y+dy, prev_pos.x+dx))
#         end
        
#         # flatten
#         positions = reshape(positions, :)
        
#         # # total update slower version 
#         # traces = [Gen.update(trace,choicemap((:steps => t => :objs => obj_id => :pos) => pos))[1] for pos in positions]

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

#         #update sprite index? todo?
        
#         #making the scores into probabilities 
#         scores_logsumexp = logsumexp(scores)
#         scores =  exp.(scores .- scores_logsumexp)

#         #oldver
#         #scores = Gen.normalize_weights(get_score.(traces))[2]
#         #scores = exp.(scores)

#         # sample the actual position
#         pos = {:steps => t => :objs => obj_id => :pos} ~ labeled_cat(positions, scores)

#         # old version: set the curr `trace` to this trace for future iterations of this loop
#         # idx = findfirst(x -> x == pos, positions)
#         # objs[obj_id] = Object(objs[obj_id].sprite, positions[idx]) 
#         # trace = traces[idx]
#     end
#     nothing 

#     #for each sprite type, propose and sample changes
#     #todo 
# end


# end # module Inference


