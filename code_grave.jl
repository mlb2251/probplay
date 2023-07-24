#code graveyard for chunks of code one writes and deletes and will never be useful but can't bear to delete them



# @gen function get_split_merge_sprite(tr)

#     num_sprite_types = tr[:init => :num_sprite_types]
    
#     H, W = size(tr[:init => :observed_image])


#     if({:split_or_merge} ~ bernoulli(n == 1 ? 1 : 0.5))
#         #splitting a sprite

#         sprite_index ~ uniform_discrete(1, num_sprite_types)

#         #not perfect 
#         new_second_sprite_index ~ uniform_discrete(sprite_index+1, num_sprite_types + 1)#is this sound? 

#         mask = tr[:init => :init_sprites => sprite_index => :mask]
#         height, width = size(mask)

#         #splitting vertically or horizontally

#         if({:vertical_or_horizontal} ~ bernoulli(height == 1 ? 0 : width ==1 ? 1 : 0.5)) #what if has width AND  height one lmao rip 
#             #splitting vertically, split point is first row of second sprite 
#             #not sounds since part could be under the other sprite, or splitting with a gap in the middle
#             split_point ~ uniform_discrete(2, height)
#         else 
#             #splitting horizontally
#             split_point ~ uniform_discrete(2, width) 
#         end

#     else 
#         #merging two sprites 

#         sprite_index1 ~ uniform_discrete(1, num_sprite_types)
#         sprite_index2 ~ uniform_discrete(sprite_index1+1, num_sprite_types) #this isn't correct
#     end 
# end


# function split_merge_sprite_involution(tr, split_merge, forward_retval, proposal_args)

#     #EVERYTHING ABOUT THIS IS NOT PERFECT IN REVERSE FIX TODO LIKE ALL THE NUMBERING STUFF AND MORE


#     #would it be better to take in one sprite as i? 
#     new_trace_choices = choicemap()
#     backward_choices = choicemap()


#     num_sprite_types = tr[:init => :num_sprite_types]
#     backward_choices[:split_or_merge] = !split_merge[:split_or_merge]

#     if split_merge[:split_or_merge]
#         #split 

#         sprite_index = split_merge[:sprite_index]
#         new_second_sprite_index = split_merge[:new_second_sprite_index]

#         backward_choices[:sprite_index1] = sprite_index
#         backward_choices[:sprite_index2] = new_second_sprite_index

#         pre_split_mask = tr[:init => :init_sprites => sprite_index => :mask]
#         vertical_or_horizontal = split_merge[:vertical_or_horizontal]

#         if vertical_or_horizontal
#             #split vertically 
#             split_point = split_merge[:split_point]
#             mask1 = pre_split_mask[1:split_point-1, :]
#             mask2 = pre_split_mask[split_point:end, :]
#             yshift, xshift = split_point - 1, 0
            
#         else 
#             #split horizontally 
#             split_point = split_merge[:split_point]
#             mask1 = pre_split_mask[:, 1:split_point-1]
#             mask2 = pre_split_mask[:, split_point:end]
#             yshift, xshift = 0, split_point - 1
#         end

#         color = tr[:init => :init_sprites => sprite_index => :color]
#         #set the first sprite 
#         new_trace_choices[(:init => :init_sprites => sprite_index => :mask)] = mask1
#         new_trace_choices[(:init => :init_sprites => sprite_index => :color)] 

#         #set the second sprite
#         new_trace_choices[(:init => :init_sprites => new_second_sprite_index => :mask)] = mask2
#         new_trace_choices[(:init => :init_sprites => new_second_sprite_index => :color)] = color

#         #for all objects of that type, split into two objects 
#         newN = tr[:init => :N]

#         for obin in 1:tr[:init => :N]
#             #is object sprite index is the sprite index
#             if tr[:init => :init_objs => obin => sprite_index] == sprite_index
#                 pos = tr[:init => :init_objs => obin => :pos]
#                 #make a second object with the second part of the sprite 
#                 newN += 1
#                 newy, newx= pos.y + yshift, pos.x + xshift
#                 tr[:init => :init_objs => newN => new_second_sprite_index => :pos] = Position(newy, newx)
#                 tr[:init => :init_objs => newN => new_second_sprite_index => :sprite_index] = new_second_sprite_index
#             end 
#         end 

#         tr[:init => :N] = newN

#     else
#         #merge 
#         sprite_index1, sprite_index2 = split_merge[:sprite_index1], split_merge[:sprite_index2]
        




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

