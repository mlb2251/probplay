#code graveyard for chunks of code one writes and deletes and will never be useful but can't bear to delete them



#doubt i need this 
# @gen function get_always_true(tr, i, hi, wi)
#     useless ~ uniform_discrete(1, 1)
# end 



#mask VERSION 2


# function mask_involution_v2(tr, always_true, forward_retval, proposal_args)
#     i, hi, wi = proposal_args
#     mask = tr[:init => :init_sprites => i => :mask]
#     height, width = size(mask)

#     backward_choices = choicemap()
#     backward_choices[:useless] = always_true[:useless]

#     new_trace_choices = choicemap()
#     #swap pixel at that index
#     mask[hi, wi] = 1 - mask[hi, wi]
#     new_trace_choices[(:init => :init_sprites => i => :mask)] = mask

#     new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
#     (new_trace, backward_choices, weight)
# end 



# @gen function get_random(tr)
#     #mask things 
#     maski ~ uniform_discrete(1, tr[:init => :num_sprite_types])	
#     mask = tr[:init => :init_sprites => maski => :mask]
#     height, width = size(mask)
#     hi ~ uniform_discrete(1, height) 
#     wi ~ uniform_discrete(1, width)	

#     #color things 
#     colori ~ uniform_discrete(1, tr[:init => :num_sprite_types])
#     rcolorshift ~ normal(0.0, 0.1)
#     gcolorshift ~ normal(0.0, 0.1)	
#     bcolorshift ~ normal(0.0, 0.1)

# end 





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

