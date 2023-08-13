#code graveyard for chunks of code one writes and deletes and will never be useful but can't bear to delete them
# "


#
#SHIFTING STUFF 
# @gen function get_drift(tr, i)
#     """non data driven"""
#     drifty ~ uniform_discrete(-10, 10)
#     driftx ~ uniform_discrete(-10, 10)
# end 



# function shift_involution(tr, drift, forward_retval, proposal_args) 
#     """moves one obj a little"""
    

#     i = proposal_args[1]

#     #@show drift 
#     new_trace_choices = choicemap()
#     backward_choices = choicemap()

#     backward_choices[:drifty] = -drift[:drifty]
#     backward_choices[:driftx] = -drift[:driftx]

#     pos = tr[:init => :init_objs => i => :pos]
#     newy = pos.y + drift[:drifty]
#     newx = pos.x + drift[:driftx]

#     newpos = Position(newy, newx)
#     new_trace_choices[(:init => :init_objs => i => :pos)] = newpos

#     new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
#     (new_trace, backward_choices, weight)
# end 



# @gen function get_random_new_color(tr, i)
#     #can't do gaus because it goes past 1 so the log pdf will get messed up 
#     r, g, b = tr[:init => :init_sprites => i => :color]

#     radius = 0.1

#     #mins are maximum of values minus radius and zero 
#     mins = [max(0, r - radius), max(0, g - radius), max(0, b - radius)]
#     maxs = [min(1, r + radius), min(1, g + radius), min(1, b + radius)]

#     rnew ~ uniform(mins[1], maxs[1])	
#     gnew ~ uniform(mins[2], maxs[2])
#     bnew ~ uniform(mins[3], maxs[3])
    
# end
# # function get_alpha(peak) #idk whats going on here lol 
# #     if peak <= 0
# #         return 0.00001
# #     elseif peak >= 1"
#         return 0.99999
#     else
#         return peak / (1 - peak)
#     end
# end

        # scores = zeros(Float64, H*W)

    # for hi in 1:H
    #     for wi in 1:W
    #         #heat score + closeness to old x, y (- distance)
    #         #can change weight about how much the distance vs heatmap matters
    #         scores[(wi-1)*H + hi] = - 5*sqrt((hi - oldy)^2 + (wi - oldx)^2) + heatmap[hi, wi]
            
            
    #     end
    # end

    # # ##uncomment to render heatmap with weighted for distance 
    # # matrixver = reshape(scores, (H, W)) 
    # # html_body(render_matrix(matrixver))

    # #@show prescores 
    # scores_logsumexp = logsumexp(scores)
    # scores =  exp.(scores .- scores_logsumexp)
        #@show old_sindicies
        #@show change_plan



        #@show "THESE ARENT FINAL "
        #@show new_trace_choices

        # # #if object of a sprite type of index greater than the one removed, change its sprite index down by ones
        # for j in 1:new_trace_choices[:init => :N]#could be more efficient to do these loops at once 
        #     si = new_trace_choices[:init => :init_objs => j => :sprite_index]
        #     if si > sprite_index
        #         new_trace_choices[:init => :init_objs => j => :sprite_index] = si - 1  
        #     end
        # end





        #old ver 



        # for noti in 1:tr[:init => :N]
        #     i = tr[:init => :N] - noti + 1 #going in reverse order to not mess up the cascades
        #     si = tr[:init => :init_objs => i => :sprite_index]
        #     @show i
        #     @show si
        #     #if object of the type to remove
        #     if si == sprite_index
        #         pos = tr[:init => :init_objs => i => :pos]
        #         # push!(positions, pos)
        #         Nremoved+=1
        #         backward_choices[:positions => Nremoved => :pos] = pos #this will have to change 
        #         if i != newN
        #             #shifty thing 
        #             shiftallobjs(i+1, newN, x -> x-1, new_trace_choices, tr)
        #         end
        #         newN -= 1
        #     #if object of a sprite type of index greater than the one removed, change its sprite index down by ones
        #     elseif si > sprite_index
                
        #         new_trace_choices[:init => :init_objs => i => :sprite_index] = si - 1 #doesn't mix well with shiftallobjs 
        #     end
        #     new_trace = update(new_trace, get_args(tr), (NoChange(),), new_trace_choices)
        # end

# # #SPLIT MERGE SPRITE 
# @gen function get_split_merge(tr)

#     N = tr[:init => :N]
    
#     H, W = size(tr[:init => :observed_image])


#     if({:split_or_merge} ~ bernoulli(N == 1 ? 1 : 0.5)) 
#         #splitting a sprite
#         #note could be same sprite type...
#         obin ~ uniform_discrete(1, N)
#         sprite_index = tr[:init => :init_objs => obin => :sprite_index]

#         #for now, new second sprite index and new obj index arent perf 
#         new_second_sprite_index ~ uniform_discrete(tr[:init => :num_sprite_types], tr[:init => :num_sprite_types])
#         #new_second_sprite_index ~ uniform_discrete(1, tr[:init => :num_sprite_types])
#         #new_obin ~ uniform_discrete(1, N+1)

#         mask = tr[:init => :init_sprites => sprite_index => :mask]
#         height, width = size(mask)


#         #splitting vertically or horizontally

#         if({:vertical_or_horizontal} ~ bernoulli(height == 1 ? 0 : width ==1 ? 1 : 0.5)) #what if has width AND  height one lmao rip 
#             #splitting vertically, split point is first row of second sprite 
#             #not sounds since part could be under the other sprite, or splitting with a gap in the middle
#             exception ~ bernoulli(0.0)
#             if height <= 1
#                 @show height
#             end 
#             split_point ~ uniform_discrete(2, height)
#         else 
#             if width <= 1
#                 exception ~ bernoulli(1.0) #hackey
#             else 
#                 exception ~ bernoulli(0.0)
#             #splitting horizontally
#                 split_point ~ uniform_discrete(2, width) 
#             end 
#         end

#     else 
#         #merging two sprites 

#         Nm1 = N - 1
        
#         obin1 ~ uniform_discrete(1, Nm1)
#         #@show N, Nm1, obin1
#         obin2 ~ uniform_discrete(obin1 + 1, N)#this isn't correct
#     end 
# end

# function get_merged_sprite(obj1, sprite1, obj2, sprite2)
#     """
#     gets sprite and relative position to obj1's position of combined sprite
#     """
#     miny, minx = min(obj1.pos.y, obj2.pos.y), min(obj1.pos.x, obj2.pos.x)
#     #@show miny, obj1.pos.y 
#     relativey1 = (miny + (-1) * obj1.pos.y)
#     relativex1 = (minx + (-1) * obj1.pos.x)
#     relativey2 = min(obj1.pos.y, obj2.pos.y) + (-1) * obj2.pos.y
#     relativex2 = min(obj1.pos.x, obj2.pos.x)  + (-1) * obj2.pos.x
#     height1, width1 = size(sprite1.mask)
#     height2, width2 = size(sprite2.mask)
#     stopi = max(obj1.pos.y + height1, obj2.pos.y + height2)
#     stopj = max(obj1.pos.x + width1, obj2.pos.x + width2)
#     height = stopi  + (-1) * miny
#     width = stopj  + (-1) * minx #check bounds 

#     #make a mask combining the two masks 
#     newmask = fill(0, (height, width))

#     #fill with relevant portions of mask1 and mask 1
#     #@show relativex1, relativey1, relativex2, relativey2


#     newmask[1-relativey1:height1-relativey1, 1-relativex1:width1-relativex1] = sprite1.mask
#     newmask[1-relativey2:height2-relativey2, 1-relativex2:width2-relativex2] = sprite2.mask
#     #second ovverides first, ok

#     #printrand = rand()
#     # if height % 5 == 0 #just random so it doesn't happen much 
#     #     html_body(render_matrix(newmask, color=3)) 
#     # end
    
#     #averages two colors 
#     newcolor = (sprite1.color + sprite2.color) / 2#this prob wont work 

#     return Sprite(newmask, newcolor), relativey1, relativex1, relativey2, relativex2
# end

# function split_merge_involution(tr, split_merge, forward_retval, proposal_args)

#     #EVERYTHING ABOUT THIS IS NOT PERFECT IN REVERSE FIX TODO LIKE ALL THE NUMBERING STUFF AND MORE


#     #would it be better to take in one sprite as i? 
#     new_trace_choices = choicemap()
#     backward_choices = choicemap()


#     #num_sprite_types = tr[:init => :num_sprite_types]
#     backward_choices[:split_or_merge] = !split_merge[:split_or_merge]

#     if split_merge[:split_or_merge]
#         #split 
#         if split_merge[:exception]
#             backward_choices[:obin1] = 1
#             backward_choices[:obin2] = 1 

#             new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
#             return (new_trace, backward_choices, weight)   
            
#         end 

#         obin = split_merge[:obin]
#         sprite_index = tr[:init => :init_objs => obin => :sprite_index]
#         new_second_sprite_index = split_merge[:new_second_sprite_index]

#         backward_choices[:obin1] = obin
#         backward_choices[:obin2] = tr[:init => :N] + 1 #fix this 

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

#         # if size(pre_split_mask)[1]%10 == 0 #just so it does it ocassionally 
#         #     html_body(render_matrix(pre_split_mask, 3))
#         #     html_body(render_matrix(mask1, 2))
#         #     html_body(render_matrix(mask2, 2))
#         # end

#         color = tr[:init => :init_sprites => sprite_index => :color]
#         #set the first sprite 
#         new_trace_choices[(:init => :init_sprites => sprite_index => :mask)] = mask1
#         new_trace_choices[(:init => :init_sprites => sprite_index => :color)] = color

#         #set the second sprite
#         new_trace_choices[(:init => :init_sprites => new_second_sprite_index => :mask)] = mask2
#         new_trace_choices[(:init => :init_sprites => new_second_sprite_index => :color)] = color

#         #shift all of the sprites if that displaced anything TODO 

#         #for all objects of that type, split into two objects 
#         newN = tr[:init => :N]

#         for obin in 1:tr[:init => :N]
#             #is object sprite index is the sprite index
#             if tr[:init => :init_objs => obin => :sprite_index] == sprite_index
#                 pos = tr[:init => :init_objs => obin => :pos]
#                 #make a second object with the second part of the sprite 
#                 newN += 1
#                 newy, newx= pos.y + yshift, pos.x + xshift
#                 #I can't just add them to the end 
#                 new_trace_choices[:init => :init_objs => newN => :pos] = Position(newy, newx)
#                 new_trace_choices[:init => :init_objs => newN => :sprite_index] = new_second_sprite_index
#             end 
#         end 

#         new_trace_choices[:init => :N] = newN

#     else
#         #merge 
#         obin1, obin2 = split_merge[:obin1], split_merge[:obin2]
#         sprite_index1, sprite_index2 = tr[:init => :init_objs => obin1 => :sprite_index], tr[:init => :init_objs => obin2 => :sprite_index]

#         newsprite, relativey1, relativex1, relativey2, relativex2 = get_merged_sprite(tr[:init => :init_objs => obin1], tr[:init => :init_sprites => sprite_index1], tr[:init => :init_objs => obin2], tr[:init => :init_sprites => sprite_index2])

#         new_trace_choices[(:init => :init_sprites => sprite_index1 => :mask)] = newsprite.mask
#         new_trace_choices[(:init => :init_sprites => sprite_index1 => :color)] = newsprite.color

#         #what to do with the second sprite? shift all the others i guess TODO 

#         #move all of the objects with those sprites 
#         N = tr[:init => :N]

#         for obin in 1:N
#             # if sprite index 1
#             if tr[:init => :init_objs => obin => :sprite_index] == sprite_index1
#                 pos = tr[:init => :init_objs => obin => :pos]
#                 newy, newx = pos.y + relativey1, pos.x + relativex1
#                 new_trace_choices[(:init => :init_objs => obin => :pos)] = Position(newy, newx)
#             end 
#             #if sprite index 2
#             if tr[:init => :init_objs => obin => :sprite_index] == sprite_index2
#                 pos = tr[:init => :init_objs => obin => :pos]
#                 newy, newx = pos.y + relativey2, pos.x + relativex2
#                 new_trace_choices[(:init => :init_objs => obin => :pos)] = Position(newy, newx)
#                 new_trace_choices[(:init => :init_objs => obin => :sprite_index)] = sprite_index1
#             end
#         end 


#         #for backward choices need new #obin, new_second_sprite_index, vertical_or_horizontal, split_point
#         backward_choices[:obin] = obin1
#         backward_choices[:exception] = false
#         backward_choices[:new_second_sprite_index] = sprite_index2 #aak 
#         backward_choices[:vertical_or_horizontal] = 1 #aaaaak 
#         backward_choices[:split_point] = 1 #something about displacement 
#     end 
#     # @show split_merge
#     # @show new_trace_choices
#     # @show backward_choices

#     new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
#     (new_trace, backward_choices, weight)    
        
# end 


# #new merge plan 
# #randomly samples 
    


















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

