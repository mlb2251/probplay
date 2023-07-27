using Gen
using Distributions
#using Random

#SPRITE STUFFFF-----------------------------------------------------------------------------------------------
@gen function add_remove_sprite_random(tr)
"""
if remove samples sprite id
if add samples sprite id, sprite mask, sprite color, list of obj ids, list of positions"""
    num_sprite_types = tr[:init => :num_sprite_types]
    if({:add_or_remove} ~ bernoulli(num_sprite_types == 1 ? 1 : 0.5)) 
        #adding 
        #make this data driven 
        C, H, W = size(tr[:init => :observed_image])
        height ~ uniform_discrete(1,H)
        width ~ uniform_discrete(1,W)
        mask ~ bernoulli_2d(0.8, height, width) #change this
        color ~ rgb_dist()
        
        N = tr[:init => :N]
        num_objs ~ poisson_plus_1(0.5)

        #todo make them not all add to the end 
        #sample position for each new object, to do make data drivennnn
        positions ~ allpositions(collect(1:num_objs), [H for _ in 1:num_objs], [W for _ in 1:num_objs])
    else 
        #removing 
        rm_sprite_index ~ uniform_discrete(1, num_sprite_types)
    end 
end 


#from heatmap, pick a badly explained place and flood fill around it everything of the same color? 


# @gen function get_new_ff_sprite(tr, heatmap)
#     #samples a place to start filling from based on heatmap
#     scores = scores_from_heatmap(heatmap)
#     place ~ categorical(scores)
#     #get the color of the place
#     observed_image = tr[:init => :observed_image]
#     @show place 
#     COME BACK TO THIS 
# end 

#STOPPED HERE




@gen function dd_add_remove_sprite_random(tr, heatmap)
    """
    if remove samples sprite id
    if add samples sprite id, sprite mask, sprite color, list of obj ids, list of positions"""
        num_sprite_types = tr[:init => :num_sprite_types]
        if({:add_or_remove} ~ bernoulli(num_sprite_types == 1 ? 1 : 0.5)) 
            #adding 
            #make this data driven 
            C, H, W = size(tr[:init => :observed_image])
            height ~ uniform_discrete(1,H)
            width ~ uniform_discrete(1,W)
            mask ~ bernoulli_2d(0.8, height, width) #change this
            color ~ rgb_dist()
            
            N = tr[:init => :N]
            num_objs ~ poisson_plus_1(0.5)
    
            #todo make them not all add to the end 
            #sample position for each new object, to do make data drivennnn
            positions ~ allpositions(collect(1:num_objs), [H for _ in 1:num_objs], [W for _ in 1:num_objs])
        else 
            #removing 
            rm_sprite_index ~ uniform_discrete(1, num_sprite_types)
        end 
    end 



function add_remove_sprite_involution(tr, add_remove_random, forward_retval, proposal_args)
    #@show add_remove_random
    #@show tr
    new_trace_choices = choicemap()
    backward_choices = choicemap()

    if add_remove_random[:add_or_remove]
        #add the new sprite , when we add the ability to add a middle sprite, make sure to deal with the cascade 
        snp1num = tr[:init => :num_sprite_types] + 1
        new_trace_choices[:init => :num_sprite_types] = snp1num

        new_trace_choices[:init => :init_sprites => snp1num => :mask] = add_remove_random[:mask]
        new_trace_choices[:init => :init_sprites => snp1num => :color] = add_remove_random[:color]
        new_trace_choices[:init => :init_sprites => snp1num => :height] = add_remove_random[:height]
        new_trace_choices[:init => :init_sprites => snp1num => :width] = add_remove_random[:width]

        #add the new objects 
        new_trace_choices[:init => :N] = tr[:init => :N] + add_remove_random[:num_objs]
        for i in 1:add_remove_random[:num_objs]
            new_trace_choices[:init => :init_objs => tr[:init => :N] + i => :pos] = add_remove_random[:positions => i => :pos]
            new_trace_choices[:init => :init_objs => tr[:init => :N] + i => :sprite_index] = tr[:init => :num_sprite_types] + 1
        end

        backward_choices[:add_or_remove] = false
        backward_choices[:rm_sprite_index] = tr[:init => :num_sprite_types] + 1

    else 
        #remove 
        sprite_index = add_remove_random[:rm_sprite_index]
        new_trace_choices[:init => :num_sprite_types] = tr[:init => :num_sprite_types] - 1

        positions = []
        #remove each object
        #remember to change N 
        # newN = tr[:init => :N]
        Nremoved = 0


        #remove all objects of that sprite type, shift all the other objects down their ois and if higher si down one
        old_sindicies = [] 
        change_plan = [] #1 means removed, negative/zero means how much to go down
        for i in 1:tr[:init => :N] 
            si = tr[:init => :init_objs => i => :sprite_index]
            #push!(old_sindicies, si)
            if si == sprite_index
                #push!(change_plan, 1)
                Nremoved += 1 
            else
                shiftallobjs(i, i, x -> x - Nremoved, new_trace_choices, tr)#just one obj actually 
                if si > sprite_index 
                    new_trace_choices[:init => :init_objs => i - Nremoved => :sprite_index] = si - 1
                end 
                #push!(change_plan, -Nremoved) #change this
            end
        end

        new_trace_choices[:init => :N] = tr[:init => :N] - Nremoved

        
        #remove the sprite
        if sprite_index != tr[:init => :num_sprite_types] #idk if this is causing the error/neccesary
            #if not the last sprite, shift all the sprites 
            shiftallsprites(sprite_index+1, tr[:init => :num_sprite_types], x -> x-1, new_trace_choices, tr)
        end
        #shiftallsprites(sprite_index+1, tr[:init => :num_sprite_types], x -> x-1, new_trace_choices, tr)
        

        backward_choices[:add_or_remove] = true
        old_mask = tr[:init => :init_sprites => sprite_index => :mask]
        backward_choices[:mask] = old_mask
        backward_choices[:height], backward_choices[:width] = size(old_mask)
        backward_choices[:color] = tr[:init => :init_sprites => sprite_index => :color]
        backward_choices[:num_objs] = Nremoved
        #backward_choices[:positions] = positions
    end 
    #@show new_trace_choices
    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    #@show new_trace
    (new_trace, backward_choices, weight)
end 





#move this. useful 
struct Uniform_not_x <: Gen.Distribution{Int} end

function Gen.random(::Uniform_not_x, x, min, max)
    #get random int between min and max not xmax
    y = rand(min:max-1)
    if y >= x
        y += 1
    end
    return y
end

function Gen.logpdf(::Uniform_not_x, y, x, min, max)
    if y == x
        return -Inf
    end 
    if y >= min && y <= max
        return -log(max-min) #log of (1/(max-min))
    else 
        return -Inf
    end
end

const uniform_not_x = Uniform_not_x()
(::Uniform_not_x)(x, min, max) = random(Uniform_not_x(), x, min, max)








# @gen function sm_helper(tr, s1, s2, relpos1i, relpos1j, relpos2i, relpos2j)
#     N = tr[:init => :N]
#     objs_with_sprite_s1 = []
#     for objin in 1:N
#         if tr[:init => :init_objs => objin => :sprite_index] == s1
#             push!(objs_with_sprite_s1, objin)
#         end
#     end 
#     s1_obj ~ uniform_discrete(objs_with_sprite_s1)#that prob won't work

#     #get obj position and obj mask 
#     s1_obj_pos = tr[:init => :init_objs => s1_obj => :pos]
#     s1_obj_mask = tr[:init => :init_sprites => s1 => :mask]
#     s1_obj_height, s1_obj_width = size(s1_obj_mask)

#     #UG HSOkfdjqksl ,fjikadsfij
    
# end

# # #SPLIT MERGE SPRITE 
# @gen function get_split_merge(tr)
#     """
#     samples if split or merge
#         if split samples: s1, s2, mask1, mask2, relpos1, relpos2, color1, color2
#         if merge samples: s1, s2, relposto1, relposto2, colorc
#     """


#     # N = tr[:init => :N]
#     num_sprite_types = tr[:init => :num_sprite_types]
#     # H, W = size(tr[:init => :observed_image])

#     if({:split_or_merge} ~ bernoulli(num_sprite_types == 1 ? 1 : 0.5)) 
#         #splitting a sprite 

#         s1 ~ uniform_discrete(1, num_sprite_types)
#         #s2 is any sprite except s1 
#         s2 ~ uniform_not_x(s1, 1, num_sprite_types)

#         # get the two relative positions 
#         #switch to not be separate later
#         #always the same for s1 for now  
#         relpos1i ~ uniform_discrete(0,0)
#         relpos1j ~ uniform_discrete(0,0)

#         #especially change this 
#         relpos2i ~ uniform_discrete(-100, 100) 
#         relpos2j ~ uniform_discrete(-100, 100)

#         #get the two masks 
#         #getting masks is so fucking hard 

        






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


# @gen function dd_get_mask(tr, i)
#     mask = tr[:init => :init_sprites => i => :mask]
#     height, width = size(mask)

#     newmask = copy(mask)
#     observed_image= tr[:init => :observed_image]

#     C, H, W = size(observed_image)

#     #samples a new mask, with each pixel flipping based on how bad it is across the image
#     newmask ~ get_




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

#SHIFTING STUFF 
@gen function dd_get_drift(tr, i, heatmap)
    #TODO change to be the heatmap at the middle of the sprite not the top left. 

    H, W = size(heatmap)

    #get old x, y
    oldx = tr[:init => :init_objs => i => :pos].x
    oldy = tr[:init => :init_objs => i => :pos].y



    scores = scores_from_heatmap(heatmap, (hi, wi) -> - 5*sqrt((hi - oldy)^2 + (wi - oldx)^2))
    
    
    
    #@show scores_logsumexp
    flatmatrixindex ~ categorical(scores)


end 


function scores_from_heatmap(heatmap, other_hiwi_func=nothing)
    H, W = size(heatmap)
    scores = zeros(Float64, H*W)

    for hi in 1:H
        for wi in 1:W
            if other_hiwi_func === nothing
                scores[(wi-1)*H + hi] = heatmap[hi, wi]
            else 
                scores[(wi-1)*H + hi] = heatmap[hi, wi] + other_hiwi_func(hi, wi)
            end
        end
    end

    # ##uncomment to render heatmap with weighted for distance 
    # matrixver = reshape(scores, (H, W)) 
    # html_body(render_matrix(matrixver))

    #@show prescores 
    scores_logsumexp = logsumexp(scores)
    scores =  exp.(scores .- scores_logsumexp)
    
    
    #@show scores_logsumexp
    return scores
end



function shift_involution(tr, drift, forward_retval, proposal_args) 
    """moves one obj a little"""
    

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

function dd_shift_involution(tr, newspot, forward_retval, proposal_args) 
    
    i = proposal_args[1]

    new_trace_choices = choicemap()
    backward_choices = choicemap()

    
    C, H, W = size(tr[:init => :observed_image])

    pos = tr[:init => :init_objs => i => :pos]
    # backward_choices[:newy] = pos.y
    # backward_choices[:newx] = pos.x
    backward_choices[:flatmatrixindex] = (pos.x-1)*H + pos.y


    flatmatrixindex = newspot[:flatmatrixindex]
    newx = floor(flatmatrixindex / H) + 1
    newy = flatmatrixindex % H

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


function shiftallsprites(min_sprite, max_sprite, shift_func, new_trace_choices, tr)
    function shiftonething(thing, i)
        new_trace_choices[(:init => :init_sprites => shift_func(i) => thing)] = tr[:init => :init_sprites => i => thing] 
    end 

    for i in min_sprite:max_sprite
        shiftonething(:mask, i)
        shiftonething(:color, i)
        shiftonething(:height, i)
        shiftonething(:width, i)
        # toshiftmask = tr[:init => :init_sprites => i => :mask]
        # toshiftcolor = tr[:init => :init_sprites => i => :color]
        # toshiftheight = tr[:init => :init_sprites => i => :height]
        # toshiftwidth = tr[:init => :init_sprites => i => :width]

        # new_trace_choices[(:init => :init_sprites => shift_func(i) => :mask)] = toshiftmask
        # new_trace_choices[(:init => :init_sprites => shift_func(i) => :color)] = toshiftcolor
        # new_trace_choices[(:init => :init_sprites => shift_func(i) => :height)] = toshiftheight
        # new_trace_choices[(:init => :init_sprites => shift_func(i) => :width)] = toshiftwidth

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
    #@show new_trace_choices
    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    #@show new_trace
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
    
