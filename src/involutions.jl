using Gen
using Distributions

#SPRITE STUFFFF-----------------------------------------------------------------------------------------------

#ADD REMOVE SPRITE TODO 
#the adding sprite won't affect anything visually hmmmm


#SPLIT MERGE SPRITE 
@gen function get_split_merge_sprite(tr)

    num_sprite_types = tr[:init => :num_sprite_types]
    
    H, W = size(tr[:init => :observed_image])


    if({:split_or_merge} ~ bernoulli(n == 1 ? 1 : 0.5))
        #splitting a sprite

        sprite_index ~ uniform_discrete(1, num_sprite_types)

        new_second_sprite_index ~ uniform_discrete(sprite_index+1, num_sprite_types + 1)#is this sound? 

        mask = tr[:init => :init_sprites => sprite_index => :mask]
        height, width = size(mask)

        #splitting vertically or horizontally

        if({:vertical_or_horizontal} ~ bernoulli(height == 1 ? 0 : width ==1 ? 1 : 0.5)) #what if has width AND  height one lmao rip 
            #splitting vertically, split point is first row of second sprite 
            split_point ~ uniform_discrete(2, height)
        else 
            #splitting horizontally
            split_point ~ uniform_discrete(2, width) 
        end

    else 
        #merging two sprites 

        sprite_index1 ~ uniform_discrete(1, num_sprite_types)
        sprite_index2 ~ uniform_discrete(sprite_index1+1, num_sprite_types) #this isn't correct
    end 
end


function split_merge_sprite_involution(tr, split_merge, forward_retval, proposal_args)
    #would it be better to take in one sprite as i? 
    new_trace_choices = choicemap()
    backward_choices = choicemap()


    num_sprite_types = tr[:init => :num_sprite_types]
    backward_choices[:split_or_merge] = !split_merge[:split_or_merge]

    if split_merge[:split_or_merge]
        #split 

        sprite_index = split_merge[:sprite_index]
        new_second_sprite_index = split_merge[:new_second_sprite_index]

        backward_choices[:sprite_index1] = sprite_index
        backward_choices[:sprite_index2] = new_second_sprite_index

        #oh shit this is harder than I thought, needs to take in a time it has seen the two sprites to decide where/how to merge them 



    #WIP TODO COME BACK DONT FORGET ABOUT ME 



    end 
end 

    
    

#SIZE STUFF
@gen function get_random_size(tr, i)
    #@show tr 
    mask = tr[:init => :init_sprites => i => :mask]
    h, w = size(mask)
    #@show mask
    
    #random grow or shrink unless the sprite is one long (grow)
    if({:hgrow_or_shrink} ~ bernoulli(h == 1 ? 1 : 0.5))
        #grow
        hchange ~ uniform_discrete(1, max(size(tr[:init => :observed_image])[2] - h, 1)) #change this? 
    else 
        #shrink 
        hchange ~ uniform_discrete(1, h -1) 
    end 


    if({:wgrow_or_shrink} ~ bernoulli(w == 1 ? 1 : 0.5))
        #grow
        wchange ~ uniform_discrete(1, max(size(tr[:init => :observed_image])[3] - w, 1)) #change this? 
    else 
        #shrink 
        wchange ~ uniform_discrete(1, w -1) 
    end 
end

function size_involution(tr, mask_stuff, forward_retval, proposal_args)
    i = proposal_args[1]
    mask = tr[:init => :init_sprites => i => :mask]
    h, w = size(mask)

    new_trace_choices = choicemap()
    backward_choices = choicemap()	

    function changedim(old, change, grow_or_shrink)
        if grow_or_shrink == 1
            return old + change
        else
            return old - change
        end
    end

    newh = changedim(h, mask_stuff[:hchange], mask_stuff[:hgrow_or_shrink])	
    neww = changedim(w, mask_stuff[:wchange], mask_stuff[:wgrow_or_shrink])

    backward_choices[:hchange], backward_choices[:wchange] = mask_stuff[:hchange], mask_stuff[:wchange]
    backward_choices[:hgrow_or_shrink], backward_choices[:wgrow_or_shrink] = 1 - mask_stuff[:hgrow_or_shrink], 1 - mask_stuff[:wgrow_or_shrink]


    #need to not fill with all ones lmao TODO 
    newmask = fill(1, (newh, neww))
    new_trace_choices[(:init => :init_sprites => i => :mask)] = newmask
    new_trace_choices[(:init => :init_sprites => i => :width)] = neww
    new_trace_choices[(:init => :init_sprites => i => :height)] = newh


    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end

    

#mask STUFF

@gen function get_random_hi_wi(tr, i)
    mask = tr[:init => :init_sprites => i => :mask]
    height, width = size(mask)
    hi ~ uniform_discrete(1, height)	
    wi ~ uniform_discrete(1, width)
end 


function mask_involution(tr, hi_wi, forward_retval, proposal_args)#what are last two for, prop[1] is i??
    i = proposal_args[1]
    
    new_trace_choices = choicemap()
    backward_choices = choicemap()

    #random pixel index is same to reverse 
    hi = hi_wi[:hi]
    wi = hi_wi[:wi]
    backward_choices[:hi] = hi
    backward_choices[:wi] = wi

    mask = copy(tr[:init => :init_sprites => i => :mask]) #will this work? 

    #swap pixel at that index 
    #@show size(mask)
    mask[hi, wi] = 1 - mask[hi, wi]

    new_trace_choices[(:init => :init_sprites => i => :mask)] = mask
    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end 


#mask VERSION 2

#doubt i need this 
# @gen function get_always_true(tr, i, hi, wi)
#     useless ~ uniform_discrete(1, 1)
# end 

function mask_involution_v2(tr, always_true, forward_retval, proposal_args)
    i, hi, wi = proposal_args
    mask = tr[:init => :init_sprites => i => :mask]
    height, width = size(mask)

    backward_choices = choicemap()
    backward_choices[:useless] = always_true[:useless]

    new_trace_choices = choicemap()
    #swap pixel at that index
    mask[hi, wi] = 1 - mask[hi, wi]
    new_trace_choices[(:init => :init_sprites => i => :mask)] = mask

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end 

            

#COLOR STUFF

@gen function get_random_new_color(tr, i)
    #can't do gaus because it goes past 1 so the log pdf will get messed up 
    r, g, b = tr[:init => :init_sprites => i => :color]

    radius = 0.1

    #mins are maximum of values minus radius and zero 
    mins = [max(0, r - radius), max(0, g - radius), max(0, b - radius)]
    maxs = [min(1, r + radius), min(1, g + radius), min(1, b + radius)]

    rnew ~ uniform(mins[1], maxs[1])	
    gnew ~ uniform(mins[2], maxs[2])
    bnew ~ uniform(mins[3], maxs[3])
    
end


@gen function dd_get_random_new_color(tr, i)
    r, g, b = tr[:init => :init_sprites => i => :color]

    mask = tr[:init => :init_sprites => i => :mask]
    height, width = size(mask)
    observed_image= tr[:init => :observed_image]
    #@show size(observed_image)
    C, H, W = size(observed_image)

    #get the average color of the sprite across all objs of that sprite 
    rsum, gsum, bsum = 0, 0, 0
    numpixels = 0

    for objind in 1:tr[:init => :N]
        if tr[:init => :init_objs => objind => :sprite_index] == i
            pos = tr[:init => :init_objs => objind => :pos]


            #ACTUALLY GOTTA ONLY CHECK STUFF IN BOUNDS 
            starti, startj, stopi, stopj = get_bounds(tr[:init => :init_objs => objind], tr[:init => :init_sprites => i], 1, H, 1, W)
            #@show starti, startj, stopi, stopj

            #add to color average if in mask 
            for hi in starti:stopi-1
                for wi in startj:stopj-1
                    if mask[hi, wi] == 1
                        rsum += observed_image[1, pos.y + hi, pos.x + wi]
                        gsum += observed_image[2, pos.y + hi, pos.x + wi]
                        bsum += observed_image[3, pos.y + hi, pos.x + wi]
                        numpixels += 1
                    end
                end
            end
        end
    end 

    #@show numpixels
    if numpixels == 0 
        ravg, gavg, bavg = 0.5, 0.5, 0.5
    else 
        ravg = rsum / numpixels
        gavg = gsum / numpixels
        bavg = bsum / numpixels
    end

    #make a better distribution
    #val = (peak/(1 - peak))
    #@dist beta_with_peak(peak) = beta((peak/(1 - peak)), 1)
    # rnew ~ beta_with_peak(ravg)
    # gnew ~ beta_with_peak(gavg)
    # bnew ~ beta_with_peak(bavg)


    function get_alpha(peak) #idk whats going on here lol 
        if peak <= 0
            return 0.00001
        elseif peak >= 1
            return 0.99999
        else
            return peak / (1 - peak)
        end
    end

    rnew ~ beta(get_alpha(ravg), 1)
    gnew ~ beta(get_alpha(gavg), 1)
    bnew ~ beta(get_alpha(bavg), 1)
    
    
end 


#I have no clue why this isn't working 
function color_involution(tr, colors, forward_retval, proposal_args)
    i = proposal_args[1]

    new_trace_choices = choicemap()
    backward_choices = choicemap()

    backward_choices[:rnew], backward_choices[:bnew], backward_choices[:gnew] = tr[:init => :init_sprites => i => :color]

    new_trace_choices[(:init => :init_sprites => i => :color)] = [colors[:rnew], colors[:gnew], colors[:bnew]]

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end 






#OBJECT STUFFFFFFFFFFFFF------------------------------------------------------------------------------------------------------


#SHIFTING STUFF 
@gen function get_drift(tr, i)
    drifty ~ uniform_discrete(-10, 10)
    driftx ~ uniform_discrete(-10, 10)
end 




function shift_involution(tr, drift, forward_retval, proposal_args) 
    i = proposal_args[1]

    #@show drift 
    new_trace_choices = choicemap()
    backward_choices = choicemap()

    backward_choices[:drifty] = -drift[:drifty]
    backward_choices[:driftx] = -drift[:driftx]

    pos = tr[:init => :init_objs => i => :pos]
    newy = pos.y + drift[:drifty]
    newx = pos.x + drift[:driftx]

    newpos = Position(newy, newx)
    new_trace_choices[(:init => :init_objs => i => :pos)] = newpos

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end 





#ADD REMOVE OBKECT 
@gen function get_add_remove_object(tr)
    n = tr[:init => :N]
    #@show n 
    num_sprite_types = tr[:init => :num_sprite_types]
    H, W = size(tr[:init => :observed_image])

    if({:add_or_remove} ~ bernoulli(n == 1 ? 1 : 0.5))
        #adding an object 
        ypos ~ uniform_discrete(1, H)
        xpos ~ uniform_discrete(1, W)
        
        sprite_index ~ uniform_discrete(1, num_sprite_types)
        add_obj_index ~ uniform_discrete(1, n+1)


    else 
        remove_obj_index ~ uniform_discrete(1, n)
        
    end 

end


function shiftallobjs(min_obj, max_obj, shift_func, new_trace_choices, tr)
    """
    edits new traces to shift all object positions and sprite indicies down or up (depending on shift_func)
    useful for adding or removing an object at a middle object index 
    """
    for i in min_obj:max_obj
        toshiftpos = tr[:init => :init_objs => i => :pos]
        toshiftspriteindex = tr[:init => :init_objs => i => :sprite_index]

        new_trace_choices[(:init => :init_objs => shift_func(i) => :pos)] = toshiftpos
        new_trace_choices[(:init => :init_objs => shift_func(i) => :sprite_index)] = toshiftspriteindex
    end 
end

function add_remove_involution(tr, add_remove_stuff, forward_retval, proposal_args)
    new_trace_choices = choicemap()
    backward_choices = choicemap()
    N = tr[:init => :N]
    backward_choices[:add_or_remove] = !add_remove_stuff[:add_or_remove]
    
    

    if add_remove_stuff[:add_or_remove]
        #add

        ind_to_add = add_remove_stuff[:add_obj_index]

        #shift all up 
        if ind_to_add < N+1
            shiftallobjs(ind_to_add, N, i -> i+1, new_trace_choices, tr)
        end
                
        new_trace_choices[(:init => :init_objs => ind_to_add => :pos)] = Position(add_remove_stuff[:ypos], add_remove_stuff[:xpos])
        new_trace_choices[(:init => :init_objs => ind_to_add => :sprite_index)] = add_remove_stuff[:sprite_index]
        backward_choices[:remove_obj_index] = ind_to_add
        new_trace_choices[(:init => :N)] = N + 1
        


    else 
        #remove 
        remove_obj_index = add_remove_stuff[:remove_obj_index]
       
        pos = tr[:init => :init_objs => remove_obj_index => :pos]
        backward_choices[:ypos], backward_choices[:xpos] = pos.y, pos.x
        backward_choices[:sprite_index] = tr[:init => :init_objs => remove_obj_index => :sprite_index]
        backward_choices[:add_obj_index] = remove_obj_index

        #shift all objects above index down by one 
        if remove_obj_index < N
            shiftallobjs(remove_obj_index+1, N, (x)->x-1, new_trace_choices, tr) 
        end 

        #remove Nth HOW 
        new_trace_choices[(:init => :init_objs => N)] = nothing 
        new_trace_choices[(:init => :N)] = N - 1
    end

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)

end 

    
    
    

#LAYERING STUFF
@gen function get_layer_swap(tr)
    n = tr[:init => :N]
    
    layer1 ~ uniform_discrete(1, n)
    layer2 ~ uniform_discrete(1, n)#could force not equal 
end

function layer_involution(tr, layer_swap, forward_retval, proposal_args)
    new_trace_choices = choicemap()
    backward_choices = choicemap()

    l1 = layer_swap[:layer1]
    l2 = layer_swap[:layer2]
    #@show l1, l2


    backward_choices[:layer1] = l1
    backward_choices[:layer2] = l2

    sprite1ind = tr[:init => :init_objs => l1 => :sprite_index]
    sprite2ind = tr[:init => :init_objs => l2 => :sprite_index]
    pos1 = tr[:init => :init_objs => l1 => :pos]
    pos2 = tr[:init => :init_objs => l2 => :pos]	

    new_trace_choices[(:init => :init_objs => l1 => :sprite_index)] = sprite2ind
    new_trace_choices[(:init => :init_objs => l2 => :sprite_index)] = sprite1ind
    new_trace_choices[(:init => :init_objs => l1 => :pos)] = pos2
    new_trace_choices[(:init => :init_objs => l2 => :pos)] = pos1

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)

end
    
