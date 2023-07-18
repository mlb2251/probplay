




# @gen function randomness(height, width)
#     hi ~ uniform_discrete(1, height)	#don;t actually want hi in the trace, look at split nerge example. one func for randomness
#     wi ~ uniform_discrete(1, width)
# end


# @gen function randomness(tr)
#     #hilist is empty list of integers 
#     hilist = []
#     #wilist is empty list of integers
#     wilist = []



#     #aaa idk if this datastructure works mappp aa 
#     for i=1:tr[:init => :num_sprite_types]
#         mask = tr[:init => :init_sprites => i => :mask]
#         height, width = size(mask)
#         hi ~ uniform_discrete(1, height)	#don;t actually want hi in the trace, look at split nerge example. one func for randomness
#         wi ~ uniform_discrete(1, width)
#         push!(hilist, hi)
#         push!(wilist, wi)
#     end
#     #return hilist, wilist
#     {hilist} = hilist
#     {wilist} = wilist
# end 

# @gen function rand_hi_wi(i, tr)
#     mask = tr[:init => :init_sprites => i => :mask]
#     height, width = size(mask)
#     hi ~ uniform_discrete(1, height)	#don;t actually want hi in the trace, look at split nerge example. one func for randomness
#     wi ~ uniform_discrete(1, width)

#     return (hi, wi)
# end 

@gen function get_random(tr)
    #mask things 
    maski ~ uniform_discrete(1, tr[:init => :num_sprite_types])	
    mask = tr[:init => :init_sprites => maski => :mask]
    height, width = size(mask)
    hi ~ uniform_discrete(1, height) 
    wi ~ uniform_discrete(1, width)	

    #color things 
    colori ~ uniform_discrete(1, tr[:init => :num_sprite_types])
    rcolorshift ~ normal(0.0, 0.1)
    gcolorshift ~ normal(0.0, 0.1)	
    bcolorshift ~ normal(0.0, 0.1)

end 






# #should do mh for each sprite separately
# all_rand_hi_wi = Map(rand_hi_wi)


# @gen function rand_hilist_wilist(tr) #collect broke it?? what? is going on/ 
#     hilist ~ all_rand_hi_wi(1:tr[:init => :num_sprite_types], [tr for _ in 1:tr[:init => :num_sprite_types]])
#     @show hilist
#     wilist ~ all_rand_hi_wi(1:tr[:init => :num_sprite_types], [tr for _ in 1:tr[:init => :num_sprite_types]])
#     @show wilist
# end 
    



# @gen function rand_hi_wi(tr)
#     for i in range (1, tr[:init => :num_sprite_types])
#         mask = tr[:init => :init_sprites => i => :mask]
#         height, width = size(mask)
#         hi ~ uniform_discrete(1, height)	#don;t actually want hi in the trace, look at split nerge example. one func for randomness
#         wi ~ uniform_discrete(1, width)
#     end



# @gen function small_mask_change(tr, sprite_index)
#     mask = tr[:init => :init_sprites => sprite_index => :mask]
#     height, width = size(mask)
#     hi ~ uniform_discrete(1, height)	#don;t actually want hi in the trace, look at split nerge example. one func for randomness
#     wi ~ uniform_discrete(1, width)
    	

#     #change mask at hi, wi
#     mask[hi, wi] = 1 - mask[hi, wi]

#     {(:init => :init_sprites => sprite_index => :mask)} = mask #need to find a way for this to be a 
# end 

#involution 
# function update_detect(tr, random_choices, retval, for_args)
#     #@show random_choices
#     new_trace_choices = choicemap()
#     backward_choices = choicemap()

#     #update num objects 
#     tr, = mh(tr, select(:N))

#     #update num sprites 
#     tr, = mh(tr, select(:num_sprite_types))

#     #update object positions
#     for i=1:tr[:init => :N]#init defined in model
#         tr, = mh(tr, select((:init => :init_objs => i => :pos))) #correct? 
#     end

#     #recolor sprites 
#     # for i=1:tr[:init => :num_sprite_types]
#     #     #@show tr[:init => :init_sprites => i => :color]
#     #     tr, = mh(tr, select((:init => :init_sprites => i => :color))) 
#     # end
    
#     colori = random_choices[:colori]
#     rcolornew = random_choices[:rcolorshift] + tr[:init => :init_sprites => colori => :color][1]
#     gcolornew = random_choices[:gcolorshift] + tr[:init => :init_sprites => colori => :color][2]
#     bcolornew = random_choices[:bcolorshift] + tr[:init => :init_sprites => colori => :color][3]

#     new_trace_choices[(:init => :init_sprites => colori => :color)] = [rcolornew, gcolornew, bcolornew]	
#     #backward_choices[(:init => :init_sprites => colori => :color)] = tr[:init => :init_sprites => colori => :color]
#     backward_choices[:colori] = colori	
#     backward_choices[:rcolorshift] = - random_choices[:rcolorshift]
#     backward_choices[:gcolorshift] = - random_choices[:gcolorshift]
#     backward_choices[:bcolorshift] = - random_choices[:bcolorshift]

#     #resprite objects
#     for i=1:tr[:init => :N]
#         tr, = mh(tr, select((:init => :init_objs => i => :sprite_index))) 
#     end

#     #reshape objects TODO
#     #@show get_args(random_choices)
    

#     # # hilist = random_choices[:hilist] 	
#     # # @show hilist
#     # # wilist = random_choices[:wilist]	
#     # for i=1:tr[:init => :num_sprite_types]
#     #     @show random_choices[:hilist -> i -> :hi]


#     #     hi = hilist[i]	
#     #     wi = wilist[i]	
#     #     mask = tr[:init => :init_sprites => i => :mask]
#     #     #backward_choices[(:init => :init_sprites => i => :mask)] = mask
#     #     backward_choices[:hilist] = hilist
#     #     backward_choices[:wilist] = wilist
#     #     mask[hi, wi] = 1 - mask[hi, wi] 
#     #     #{(:init => :init_sprites => i => :mask)} = mask 
#     #     new_trace_choices[(:init => :init_sprites => i => :mask)] = mask

#     #     #tr = mh(tr, small_mask_change, (tr, i,))
#     #     #tr, = mh(tr, select((:init => :init_sprites => i => :mask))) 
#     # end


#     #reshape objects new way 
#     backward_choices[:maski] = random_choices[:maski]
#     backward_choices[:hi] = random_choices[:hi]
#     backward_choices[:wi] = random_choices[:wi]

#     maski = random_choices[:maski]
#     hi = random_choices[:hi]
#     wi = random_choices[:wi]

#     mask = tr[:init => :init_sprites => maski => :mask]
#     mask[hi, wi] = 1 - mask[hi, wi]
#     new_trace_choices[(:init => :init_sprites => maski => :mask)] = mask




#     #add/delete TODO

#     #split/merge TODO
    
#     new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
#     (new_trace, backward_choices, weight)
# end


#SPRITE STUFFFF


#SIZE STUFF
@gen function get_random_size(tr, i)
    #@show tr 
    mask = tr[:init => :init_sprites => i => :mask]
    h, w = size(mask)
    #@show mask
    
    #random grow or shrink unless the sprite is one long (grow)
    if({:hgrow_or_shrink} ~ bernoulli(h == 1 ? 1 : 0.3))
        #grow
        hchange ~ uniform_discrete(1, max(size(tr[:init => :observed_image])[2] - h, 1)) #change this? 
    else 
        #shrink 
        hchange ~ uniform_discrete(1, h -1) 
    end 


    if({:wgrow_or_shrink} ~ bernoulli(w == 1 ? 1 : 0.3))
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

function color_involution(tr, colors, forward_retval, proposal_args)
    i = proposal_args[1]

    new_trace_choices = choicemap()
    backward_choices = choicemap()

    backward_choices[:rnew], backward_choices[:bnew], backward_choices[:gnew] = tr[:init => :init_sprites => i => :color]

    new_trace_choices[(:init => :init_sprites => i => :color)] = [colors[:rnew], colors[:gnew], colors[:bnew]]

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end 


#COLOR VERSION 2
#function color_set(tr)




#OBJECT STUFFFFFFFFFFFFF


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
@gen function get_add_remove(tr)
    n = tr[:init => :N]
    #@show n 
    num_sprite_types = tr[:init => :num_sprite_types]
    H, W = size(tr[:init => :observed_image])

    if({:add_or_remove} ~ bernoulli(n == 1 ? 1 : 0.3))
        #adding an object 
        ypos ~ uniform_discrete(1, H)
        xpos ~ uniform_discrete(1, W)
        
        sprite_index ~ uniform_discrete(1, num_sprite_types)
        #add_obj_index ~ uniform_discrete(1, n+1)


    else 
        remove_obj_index ~ uniform_discrete(1, n)
        
    end 

end


function add_remove_involution(tr, add_remove_stuff, forward_retval, proposal_args)
    #TODO fix not perfect undoing because the ordering will be wrong 
    #@show add_remove_stuff
    new_trace_choices = choicemap()
    backward_choices = choicemap()
    N = tr[:init => :N]
    backward_choices[:add_or_remove] = !add_remove_stuff[:add_or_remove]
    

    if add_remove_stuff[:add_or_remove]
        #add
        new_trace_choices[(:init => :init_objs => N + 1 => :pos)] = Position(add_remove_stuff[:ypos], add_remove_stuff[:xpos])
        
        new_trace_choices[(:init => :init_objs => N + 1 => :sprite_index)] = add_remove_stuff[:sprite_index]

        backward_choices[:remove_obj_index] = N + 1
        new_trace_choices[(:init => :N)] = N + 1
        

    else 
        #remove 
        #@show add_remove_stuff[:remove_obj_index]
        remove_obj_index = add_remove_stuff[:remove_obj_index]
        #shift all objects above index down by one 
        pos = tr[:init => :init_objs => remove_obj_index => :pos]
        backward_choices[:ypos], backward_choices[:xpos] = pos.y, pos.x
        backward_choices[:sprite_index] = tr[:init => :init_objs => remove_obj_index => :sprite_index]
        if remove_obj_index < N
            for n in remove_obj_index+1:N
                toshiftpos = tr[:init => :init_objs => n => :pos]
                toshiftspriteindex = tr[:init => :init_objs => n => :sprite_index]

                new_trace_choices[(:init => :init_objs => n - 1 => :pos)] = toshiftpos
                new_trace_choices[(:init => :init_objs => n - 1 => :sprite_index)] = toshiftspriteindex
            end 
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
    
