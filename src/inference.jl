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


# function test_one_involution(frame)
#     (C, H, W) = size(frame)
#     (cluster, objs, sprites) = process_first_frame(frame)

#     cm = build_init_obs(H,W, sprites, objs, frame)

#     tr = generate(model, (H, W, 1), cm)[1]


#     for repeat in 1:100#0
#         #THING U WANNA TEST HERE 
#         #tr, accepted = mh(tr, get_split_merge, (), split_merge_involution)
#         #tr = total_update(tr)
#         #@show tr
        
#         # heatmap = logpdfmap(ImageLikelihood(), tr[:init => :observed_image], render_trace_frame(tr, 0), 0.1)
#         # tr, accepted = mh(tr, dd_add_remove_sprite_random, (heatmap,), dd_add_remove_sprite_involution)
#         # if accepted
#         #     print("added/removed sprite")
#         # end
#         tr, accepted = mh(tr, get_add_remove_object, (), add_remove_involution)
#         if accepted
#             print("added/removed sprite")
#         end



#         # if accepted
#         #     print("split/merge")
#         # end
#         #render trace
        
#         if repeat % 10 == 0

#             html_body(html_img(draw(H, W, tr[:init => :init_state => :init_objs], tr[:init => :init_sprites])))
#         end 
#     end 
# render();
# end 


function total_update(tr, stats)
    #do one pass of making a heatmap
    
    heatmap = logpdfmap(ImageLikelihood(), tr[:init => :observed_image], render_trace_frame(tr, 0), 0.1)
    

    #sprite proposals 

    #add/remove sprite 
    # orig_M = tr[:init => :num_sprites]
    for _ in 1:1
        tr, accepted = Gen.mh_tracked(tr, add_remove_sprite_random, (), add_remove_sprite_involution; stats=stats)
    end
    # if accepted
    #     @assert orig_M != tr[:init => :num_sprites]
    #     if orig_M > tr[:init => :num_sprites]
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
    

    for i=1:tr[:init => :num_sprites] #some objects need more attention. later don't make this just loop through, sample i as well
    

        #recolor involution 
        # tr, accepted = mh(tr, dd_get_random_new_color, (i,), color_involution)
        tr, accepted = mh(tr, select(:init => :init_sprites => i => :color => :r))
        tr, accepted = mh(tr, select(:init => :init_sprites => i => :color => :g))
        tr, accepted = mh(tr, select(:init => :init_sprites => i => :color => :b))

        #resize involution 
        tr, accepted = Gen.mh_tracked(tr, get_random_size, (i,), size_involution; stats=stats)

     #reshape involution 
        for _ in 1:10 #doing more of these 
            tr, accepted = Gen.mh_tracked(tr, get_random_hi_wi, (i,), mask_involution; stats=stats)
        end 
    end 

    #object proposals 

    #add/remove object involution 
    # orig_N = tr[:init => :init_state => :N]
    tr, accepted = Gen.mh_tracked(tr, get_add_remove_object, (), add_remove_involution; stats=stats)
    # if accepted
    #     @assert orig_N != tr[:init => :init_state => :N]
    #     if orig_N > tr[:init => :init_state => :N]
    #         print("added obj")
    #     else
    #         print("removed obj")
    #     end
    # end

    #relayer order objects
    tr, accepted = Gen.mh_tracked(tr, get_layer_swap, (), layer_involution; stats=stats)

    for i=1:tr[:init => :init_state => :N]
        #shift objects involution 

        tr, accepted = Gen.mh_tracked(tr, dd_get_drift, (i, heatmap,), dd_shift_involution; stats=stats) 

        #resprite object involution ok. 
        tr, accepted = mh(tr, select((:init => :init_state => :init_objs => i => :sprite_index)))
    end 

    tr
end 


function mh_first_frame(traces; steps=1000, step_chunk=50, final_T=nothing)
    #run update detect a bunch TODO 
    #@show num_particles, steps, step_chunk
    (H,W,T) = get_args(traces[1])
    num_particles = length(traces)

    #something like this but edit
    # init_obs = choicemap(
        # (:init => :observed_image, frame),
        
        #have perfect build init obs here for everything else and one by one delete 
        #(:init => :init_state => :N, length(objs)),
        #(:init => :num_sprites, length(sprites)),
    # )

    # (cluster, objs, sprites) = process_first_frame(frame)
    # init_obs = build_init_obs(H,W, sprites, objs, frame)


    # traces = [generate(model, (H, W, 1), init_obs)[1] for _ in 1:num_particles]

    render_table = fill("", num_particles*2, 1)
    for i in 1:num_particles
        render_table[i*2,1] = "Particle $i"
    end

    detailed_table_header = ["" "Rendered" "Bounding Boxes" "Sprites" #= "Coloring by object" "Coloring by sprite" "logpdf heatmap" =# "(actual - rendered)" "Info" "Involutions"]


    # one detailed table per particle
    detailed_tables = [detailed_table_header for _ in 1:num_particles]



    all_stats = [Dict{Symbol, Gen.MHStats}() for _ in 1:num_particles]

    println("Threads: ", Threads.nthreads())
    @show IMG_VAR

    elapsed=@elapsed for i in 1:steps
        if i % step_chunk == 1 || i == steps
            println("update: $i")

            # update particle over time
            col = String[]
            for tr in traces
                env = env_of_trace(tr)
                push!(col, html_img(draw(H, W, env.state.objs, env.sprites)))
                push!(col, "step=$i<br>N=$(length(env.state.objs))<br>sprites=$(length(env.sprites))")
            end
            render_table = hcat(render_table, col)

            # update detailed tables
            for (j,tr) in enumerate(traces)
                detailed_tables[j] = vcat(detailed_tables[j], permutedims(detailed_trace_row(tr, "Particle $j @ step=$i"; stats=all_stats[j], final_T=final_T)))
            end

        end

        #=Threads.@threads=# for j in 1:num_particles
            traces[j] = total_update(traces[j], all_stats[j])
        end
    end

    secs_per_step = round(elapsed/steps,sigdigits=3)
    fps = round(1/secs_per_step,sigdigits=3)
    time_str = "MCMC runtime: $(round(elapsed,sigdigits=3))s; $(secs_per_step)s/step; $fps step/s ($num_particles particles))"
    println(time_str)

    #TODO MAKE 2 other tables so this isn't uglyy 

    result_table = fill("", num_particles + 1, length(detailed_table_header))
    result_table[1, :] = detailed_table_header

    for (i,tr) in enumerate(traces)
        result_table[i+1,:] = detailed_trace_row(tr, "Particle $i @ step=$steps"; stats=all_stats[i], final_T=final_T)
    end 
    
    html_body("<h1>Results</h1>")
    html_body(html_table(result_table))
    html_body("<h1>Particles Over Time</h1>")
    html_body(html_table(render_table))
    html_body(time_str)
    html_body("<h1>Particles Over Time (Detailed)</h1>")

    table_of_tables = permutedims(["<h2>Particle $i</h2>\n" * html_table(table) for (i,table) in enumerate(detailed_tables)])
    html_body(html_table(table_of_tables; table_attrs="style=\"border: 0px;\"", tr_attrs="style=\"border: 0px;\"", td_attrs="style=\"border: 0px; vertical-align:top; padding-right:20px\""))

 
end




function detailed_trace_row(tr, rowname; stats=nothing, final_T=nothing)

    (H,W,T) = get_args(tr)
    env = env_of_trace(tr)
    sprites = env.sprites
    objs = env.state.objs
    # rendered = html_img(render_trace_frame(tr, 0))
    rendered = html_gif(render_trace(tr); pad_to=final_T)
    bboxes = html_img(draw_bboxes(render_trace_frame(tr, 0), objs, sprites, 1, H, 1, W))
    # objcoloring = html_img(color_labels(obj_frame(tr[:init => :init_state => :init_objs], tr[:init => :init_sprites], H, W))[1])
    # spritecoloring = html_img(color_labels(sprite_frame(tr[:init => :init_state => :init_objs], tr[:init => :init_sprites], H, W))[1])
    # heatmap = render_heatmap(logpdfmap(ImageLikelihood(), tr[:init => :observed_image], render_trace_frame(tr, 0), 0.1))
    diff = html_img(img_diff(tr[:init => :observed_image], render_trace_frame(tr, 0)))

    sprite_imgs = map(enumerate(sprites)) do (i,sprite)
        show_sprite(tr,i)
    end
    info = "score=$(round(get_score(tr),sigdigits=6))<br>N=$(length(objs))<br>sprites=$(length(sprites))"

    show_stats = ""
    if stats !== nothing
        for key in sort(collect(keys(stats)))
            show_stats != "" && (show_stats *= "<br>")
            substats = stats[key]
            show_stats *= "$(key):&nbsp;$(substats.accepted)/$(substats.total)&nbsp;[inf=$(substats.inf);&nbsp;nan=$(substats.nan)]"
        end
    end
    row = [rowname, rendered, bboxes, join(sprite_imgs,"<br>"), #=objcoloring, spritecoloring, heatmap, =# diff, info, show_stats]
    row
end

function show_sprite(tr,i)
    env = env_of_trace(tr)
    (C,H,W) = size(tr[:init => :observed_image])
    sprite = env.sprites[i]
    colors = get_colors(length(env.sprites))
    ystop,xstop = size(sprite.mask)
    canvas = zeros(Float64, C, ystop, W) # make canvas at full screen width because we scale our rendered html images based on width
    canvas[:, :, xstop + 1:end] .= 1. # set the righthand side of the image to all white
    # fill in actual sprite
    canvas[:, 1:ystop, 1:xstop] .= ifelse.(reshape(sprite.mask, 1, size(sprite.mask)...), sprite.color, [0.,0.,0.])

    # add an alpha blended border
    alpha = 0.5
    canvas[:, 1:ystop, 1] .= alpha_blend.(colors[i], canvas[:, 1:ystop, 1], alpha)
    xstop != 1 && (canvas[:, 1:ystop, xstop] .= alpha_blend.(colors[i], canvas[:, 1:ystop, xstop], alpha))
    canvas[:, 1, 2:xstop-1] .= alpha_blend.(colors[i], canvas[:, 1, 2:xstop-1], alpha)
    ystop != 1 && (canvas[:, ystop, 2:xstop-1] .= alpha_blend.(colors[i], canvas[:, ystop, 2:xstop-1], alpha))

    img = html_img(canvas)
    num_objs = sum([obj.sprite_index == i for obj in env.state.objs])
    "$(num_objs)x$img"
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


            # for Hindex in 1:bigH-smallH+1
            #     for Windex in 1:bigW-smallW+1
            #         submask = bigmask[Hindex:Hindex+smallH-1, Windex:Windex+smallW-1] #check indicies here 
            #         mask_diff = get_mask_diff(submask, smallmask, sprite.color, color[c])
            #         # @show mask_diff	

            #         if mask_diff < 0.2
            #             println("holy shit this actually worked")
            #             newsprite = false

            #             if newbigger
            #                 #fixing old sprite type #todo should also fix its pos 
            #                 sprites[i] = Sprite(bigmask,sprite.color)
            #                 object = Object(i, Vec(smallest_y[c], smallest_x[c]))
            #                 push!(objs, object)
            #             else 
            #                 #new sprite is old just starting at diff index
            #                 object = Object(i, Vec(max(smallest_y[c]-Hindex, 1), max(smallest_x[c]-Windex, 1))) #looks weird when <0
                            
            #                 push!(objs, object)
            #             end 
            #             break # wont this jump to a weird spot
                        
            #         end
            #     end 
            # end
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

function build_init_obs(H,W, sprites, objs, first_frame)
    init_obs = choicemap(
        (:init => :observed_image, first_frame),
        (:init => :init_state => :N, length(objs)),
        (:init => :num_sprites, length(sprites)),
    )

    for (i,obj) in enumerate(objs)
        # @show obj.sprite_index
        @assert 0 < obj.pos.x <= W && 0 < obj.pos.y <= H
        init_obs[(:init => :init_state => :init_objs => i => :pos)] = obj.pos
        init_obs[(:init => :init_state => :init_objs => i => :sprite_index)] = obj.sprite_index #anything not set here it makes a random choice about

        #EDIT THIS 
        init_obs[:init => :init_sprites => obj.sprite_index => :mask] = sprites[obj.sprite_index].mask # => means go into subtrace, here initializing subtraces, () are optional. => means pair!!
        init_obs[:init => :init_sprites => obj.sprite_index => :color => :r] = sprites[obj.sprite_index].color[1]
        init_obs[:init => :init_sprites => obj.sprite_index => :color => :g] = sprites[obj.sprite_index].color[2]
        init_obs[:init => :init_sprites => obj.sprite_index => :color => :b] = sprites[obj.sprite_index].color[3]
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
function particle_filter(num_particles::Int, observed_images::Array{Float64,4}, num_samples::Int; mh_steps_init=800, mh_steps=10, floodfill=true, perception_mh=false)
    C,H,W,T = size(observed_images)

    # construct initial observations

    html_body("<script>tMax = $T;</script>")
    html_body("<p>C: $C, H: $H, W: $W, T: $T</p>")
    html_body("<h2>Observations</h2>", html_gif(observed_images))

    init_obs = if floodfill
        (cluster, objs, sprites) = process_first_frame(observed_images[:,:,:,1])
         build_init_obs(H,W, sprites, objs, observed_images[:,:,:,1])
    else
        choicemap((:init => :observed_image, observed_images[:,:,:,1]))
    end

    state = pf_initialize(model, (H,W,1), init_obs, num_particles)

    perception_mh && mh_first_frame(state.traces; steps=mh_steps_init, final_T=T)

    # steps
    elapsed=@elapsed for t in 1:T-1
        @show t
        # @show state.log_weights, weights
        # maybe_resample!(state, ess_threshold=num_particles/2, verbose=true)
        obs = choicemap((:steps => t => :observed_image, observed_images[:,:,:,t+1]))

        # take SMCP3 step
        @time pf_update!(state, (H,W,t+1), (NoChange(),NoChange(),UnknownChange()),
            obs, SMCP3Update(
                fwd_proposal_naive,
                bwd_proposal_naive,
                (obs,),
                (obs,),
                false, # check are inverses
            ))        

        


        for i in 1:num_particles

            #COMEBACK 
            tr = state.traces[i]
            # @show tr 
            #positions is a vector across times across objects giving their positions 
            positions = Array{Vec}(undef, t, length(env_of_trace(tr).state.objs))
            for time in 1:t
                stateobjs = tr[:steps => t].objs
                for obj_id in 1:length(stateobjs)
                    positions[time, obj_id] = stateobjs[obj_id].pos
                end
            end 
            #@show positions  


            for obj_id in 1:length(env_of_trace(tr).state.objs)

                # #uncomment this, mh choosing which step function 
                # tr, accept = mh(tr, select(:init => :init_state => :init_objs => obj_id => :step_of_obj))
                

                #mh comparing projected position from resampled step function to smc percieved positions 
                obj_positions = positions[:, obj_id]#across all time steps ==
                @show obj_positions
                projected_position_vec = [] 

                for sample in 1:3
                    #i can just put all of this in the fwd proposal 
                    #possible_sample = {(:possible_sample, obj_id, sample)} ~ code_prior(0, Yay)
                    #shoot this isn't a gen func 

                    #@show possible_sample 
                    #turn sampled code into positions (can just do one run or many )

                    #add projected postitions into projected position vec 
                    
                end 

                #sample projected position vec according to similarity to obj positions, adding across all positions at t to t+1 

                #have and set the sampled code 




                #mh based on rendering of resampled step function
                for _ in 1:500
                    tr, accept = mh(tr, select(:init => :cfuncs => obj_id => :sampled_code))
                    if accept 
                        # observed_image = tr[:steps => t => :observed_image]
                        # rendered = render_trace_frame(tr, t)
                        #println(Gen.logpdf(image_likelihood, observed_image, rendered, 0.1), Gen.get_submap(get_choices(tr), (:init => (:sampled_code, obj_id))))
                        
                    end  
                end 
                


                #doing a shifting involution on each attribute 
                for attr_id in 1:1 #just velocity for now 
                    attr_address = (:init => :init_state => :init_objs => obj_id => :attrs => attr_id => :attr)
                    for _ in 1:100
                        tr, accept = mh(tr, variable_shift_randomness, (attr_address,), variable_shift_involution)
                    end 
                end

            end 
            state.traces[i] = tr
           
        end 
       

        html_body("<h2>Reconstructed Images</h2>")
        table = fill("", 3, length(state.traces))
        for (i,trace) in enumerate(state.traces)
            table[1,i] = "Particle $i ($(round(state.log_weights[i],sigdigits=4)))"
            rendered = render_trace(trace)
            table[2,i] = html_gif(rendered, pad_to=T);
            table[3,i] = html_gif(img_diff(rendered, observed_images[:,:,:,1:t+1]), pad_to=T);
            for obj_id in 1:length(env_of_trace(trace).state.objs)
                
                html_body(trace[:init => :cfuncs => obj_id => :sampled_code], "<br>")
            end    
        end

        perception_mh && mh_first_frame(state.traces; steps=mh_steps, final_T=T)

        html_body("<h2>Reconstructed Images</h2>")
        table = fill("", 3, length(state.traces))
        for (i,trace) in enumerate(state.traces)
            table[1,i] = "Particle $i ($(round(state.log_weights[i],sigdigits=4)))"
            rendered = render_trace(trace)
            table[2,i] = html_gif(rendered, pad_to=T);
            table[3,i] = html_gif(img_diff(rendered, observed_images[:,:,:,1:t+1]), pad_to=T);
            for obj_id in 1:length(env_of_trace(trace).state.objs)
                html_body(trace[:init => :cfuncs => obj_id => :sampled_code], "<br>")
            end  
        end

        html_body(html_table(table))
    end


    (_, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_normalized_weights)

    # print and render results

    secs_per_step = round(elapsed/(T-1),sigdigits=3)
    fps = round(1/secs_per_step,sigdigits=3)
    time_str = "particle filter runtime: $(round(elapsed,sigdigits=3))s; $(secs_per_step)s/frame; $fps fps ($num_particles particles))"
    println(time_str)

    # types = map(i -> objs[i].sprite_index, cluster);
    # html_body("<h2>First Frame Processing</h2>",
    # html_table(["Observation"                       "Objects"                       "Types";
    #          html_img(observed_images[:,:,:,1])  html_img(color_labels(cluster)[1])  html_img(color_labels(types)[1])
    # ]))




    # types = map(i -> objs[i].sprite_index, cluster);
    # html_body("<h2>First Frame Processing</h2>",
    # html_table(["Observation"                       "Objects"                       "Types";
    #          html_img(observed_images[:,:,:,1])  html_img(color_labels(cluster)[1])  html_img(color_labels(types)[1])
    # ]))
    # html_body("<h2>Reconstructed Images</h2>")

    # table = fill("", 3, length(state.traces))
    # for (i,trace) in enumerate(state.traces)
    #     table[1,i] = "Particle $i ($(round(weights[i],sigdigits=4)))"
    #     rendered = render_trace(trace)
    #     table[2,i] = html_gif(rendered);
    #     table[3,i] = html_gif(img_diff(rendered, observed_images));

    # end

    # html_body(html_table(table))
    html_body(time_str)
    # #showing final code 
    # for obj_id in 1:4
    #     html_body(state_traces[T][:init => :(sampled_code, obj_id)])
    # end 

    
    return sample_unweighted_traces(state, num_samples)
end

# function rejuv(trace)
#     return trace, 0.
# end

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
                rendered = draw(H, W, state.objs, curr_env.sprites) #positions in here 
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
        



        #NOW TO DO RESAMPLE THE STEP FUNCITON GIVEN THE POSITIONS 

        
        #TODO given these positions, sample function that maps from pairs of positions for synthesis problem 
        #need a distribution, maybe use all the positions from the samples above
        #intuition will alternate between picking better function and better positions 
        # @show positions 


        # obj_positions = [] 
        # for time in 1:t 
        #     @show trace_updates

        # end 




    end



    # @show [curr_env.step_of_obj[i] for i in eachindex(curr_env.state.objs)]

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


