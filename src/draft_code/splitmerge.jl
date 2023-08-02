#Not currently functional, but splitmerge progress stored here


@gen function sm_helper(tr, s1, s2, relpos1i, relpos1j, relpos2i, relpos2j)
    N = tr[:init => :N]
    objs_with_sprite_s1 = []
    for objin in 1:N
        if tr[:init => :init_objs => objin => :sprite_index] == s1
            push!(objs_with_sprite_s1, objin)
        end
    end 
    s1_obj ~ uniform_discrete(objs_with_sprite_s1)#that prob won't work

    #get obj position and obj mask 
    s1_obj_pos = tr[:init => :init_objs => s1_obj => :pos]
    s1_obj_mask = tr[:init => :init_sprites => s1 => :mask]
    s1_obj_height, s1_obj_width = size(s1_obj_mask)

    #UG HSOkfdjqksl ,fjikadsfij
    
end

# #SPLIT MERGE SPRITE 
@gen function get_split_merge(tr)
    """
    samples if split or merge
        if split samples: s1, s2, mask1, mask2, relpos1, relpos2, color1, color2
        if merge samples: s1, s2, relposto1, relposto2, colorc
    """


    # N = tr[:init => :N]
    num_sprite_types = tr[:init => :num_sprite_types]
    # H, W = size(tr[:init => :observed_image])

    if({:split_or_merge} ~ bernoulli(num_sprite_types == 1 ? 1 : 0.5)) 
        #splitting a sprite 

        s1 ~ uniform_discrete(1, num_sprite_types)
        #s2 is any sprite except s1 
        s2 ~ uniform_not_x(s1, 1, num_sprite_types)

        # get the two relative positions 
        #switch to not be separate later
        #always the same for s1 for now  
        relpos1i ~ uniform_discrete(0,0)
        relpos1j ~ uniform_discrete(0,0)

        #especially change this 
        relpos2i ~ uniform_discrete(-100, 100) 
        relpos2j ~ uniform_discrete(-100, 100)

        #get the two masks 
        #getting masks is so fucking hard 

        






        #splitting a sprite
        #note could be same sprite type...
        obin ~ uniform_discrete(1, N)
        sprite_index = tr[:init => :init_objs => obin => :sprite_index]

        #for now, new second sprite index and new obj index arent perf 
        new_second_sprite_index ~ uniform_discrete(tr[:init => :num_sprite_types], tr[:init => :num_sprite_types])
        #new_second_sprite_index ~ uniform_discrete(1, tr[:init => :num_sprite_types])
        #new_obin ~ uniform_discrete(1, N+1)

        mask = tr[:init => :init_sprites => sprite_index => :mask]
        height, width = size(mask)


        #splitting vertically or horizontally

        if({:vertical_or_horizontal} ~ bernoulli(height == 1 ? 0 : width ==1 ? 1 : 0.5)) #what if has width AND  height one lmao rip 
            #splitting vertically, split point is first row of second sprite 
            #not sounds since part could be under the other sprite, or splitting with a gap in the middle
            exception ~ bernoulli(0.0)
            if height <= 1
                @show height
            end 
            split_point ~ uniform_discrete(2, height)
        else 
            if width <= 1
                exception ~ bernoulli(1.0) #hackey
            else 
                exception ~ bernoulli(0.0)
            #splitting horizontally
                split_point ~ uniform_discrete(2, width) 
            end 
        end

    else 
        #merging two sprites 

        Nm1 = N - 1
        
        obin1 ~ uniform_discrete(1, Nm1)
        #@show N, Nm1, obin1
        obin2 ~ uniform_discrete(obin1 + 1, N)#this isn't correct
    end 
end

function get_merged_sprite(obj1, sprite1, obj2, sprite2)
    """
    gets sprite and relative position to obj1's position of combined sprite
    """
    miny, minx = min(obj1.pos.y, obj2.pos.y), min(obj1.pos.x, obj2.pos.x)
    #@show miny, obj1.pos.y 
    relativey1 = (miny + (-1) * obj1.pos.y)
    relativex1 = (minx + (-1) * obj1.pos.x)
    relativey2 = min(obj1.pos.y, obj2.pos.y) + (-1) * obj2.pos.y
    relativex2 = min(obj1.pos.x, obj2.pos.x)  + (-1) * obj2.pos.x
    height1, width1 = size(sprite1.mask)
    height2, width2 = size(sprite2.mask)
    stopi = max(obj1.pos.y + height1, obj2.pos.y + height2)
    stopj = max(obj1.pos.x + width1, obj2.pos.x + width2)
    height = stopi  + (-1) * miny
    width = stopj  + (-1) * minx #check bounds 

    #make a mask combining the two masks 
    newmask = fill(0, (height, width))

    #fill with relevant portions of mask1 and mask 1
    #@show relativex1, relativey1, relativex2, relativey2


    newmask[1-relativey1:height1-relativey1, 1-relativex1:width1-relativex1] = sprite1.mask
    newmask[1-relativey2:height2-relativey2, 1-relativex2:width2-relativex2] = sprite2.mask
    #second ovverides first, ok

    #printrand = rand()
    # if height % 5 == 0 #just random so it doesn't happen much 
    #     html_body(render_matrix(newmask, color=3)) 
    # end
    
    #averages two colors 
    newcolor = (sprite1.color + sprite2.color) / 2#this prob wont work 

    return Sprite(newmask, newcolor), relativey1, relativex1, relativey2, relativex2
end

function split_merge_involution(tr, split_merge, forward_retval, proposal_args)

    #EVERYTHING ABOUT THIS IS NOT PERFECT IN REVERSE FIX TODO LIKE ALL THE NUMBERING STUFF AND MORE


    #would it be better to take in one sprite as i? 
    new_trace_choices = choicemap()
    backward_choices = choicemap()


    #num_sprite_types = tr[:init => :num_sprite_types]
    backward_choices[:split_or_merge] = !split_merge[:split_or_merge]

    if split_merge[:split_or_merge]
        #split 
        if split_merge[:exception]
            backward_choices[:obin1] = 1
            backward_choices[:obin2] = 1 

            new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
            return (new_trace, backward_choices, weight)   
            
        end 

        obin = split_merge[:obin]
        sprite_index = tr[:init => :init_objs => obin => :sprite_index]
        new_second_sprite_index = split_merge[:new_second_sprite_index]

        backward_choices[:obin1] = obin
        backward_choices[:obin2] = tr[:init => :N] + 1 #fix this 

        pre_split_mask = tr[:init => :init_sprites => sprite_index => :mask]
        vertical_or_horizontal = split_merge[:vertical_or_horizontal]

        if vertical_or_horizontal
            #split vertically 
            split_point = split_merge[:split_point]
            mask1 = pre_split_mask[1:split_point-1, :]
            mask2 = pre_split_mask[split_point:end, :]
            yshift, xshift = split_point - 1, 0
            
        else 
            #split horizontally 
            split_point = split_merge[:split_point]
            mask1 = pre_split_mask[:, 1:split_point-1]
            mask2 = pre_split_mask[:, split_point:end]
            yshift, xshift = 0, split_point - 1
        end

        # if size(pre_split_mask)[1]%10 == 0 #just so it does it ocassionally 
        #     html_body(render_matrix(pre_split_mask, 3))
        #     html_body(render_matrix(mask1, 2))
        #     html_body(render_matrix(mask2, 2))
        # end

        color = tr[:init => :init_sprites => sprite_index => :color]
        #set the first sprite 
        new_trace_choices[(:init => :init_sprites => sprite_index => :mask)] = mask1
        new_trace_choices[(:init => :init_sprites => sprite_index => :color)] = color

        #set the second sprite
        new_trace_choices[(:init => :init_sprites => new_second_sprite_index => :mask)] = mask2
        new_trace_choices[(:init => :init_sprites => new_second_sprite_index => :color)] = color

        #shift all of the sprites if that displaced anything TODO 

        #for all objects of that type, split into two objects 
        newN = tr[:init => :N]

        for obin in 1:tr[:init => :N]
            #is object sprite index is the sprite index
            if tr[:init => :init_objs => obin => :sprite_index] == sprite_index
                pos = tr[:init => :init_objs => obin => :pos]
                #make a second object with the second part of the sprite 
                newN += 1
                newy, newx= pos.y + yshift, pos.x + xshift
                #I can't just add them to the end 
                new_trace_choices[:init => :init_objs => newN => :pos] = Position(newy, newx)
                new_trace_choices[:init => :init_objs => newN => :sprite_index] = new_second_sprite_index
            end 
        end 

        new_trace_choices[:init => :N] = newN

    else
        #merge 
        obin1, obin2 = split_merge[:obin1], split_merge[:obin2]
        sprite_index1, sprite_index2 = tr[:init => :init_objs => obin1 => :sprite_index], tr[:init => :init_objs => obin2 => :sprite_index]

        newsprite, relativey1, relativex1, relativey2, relativex2 = get_merged_sprite(tr[:init => :init_objs => obin1], tr[:init => :init_sprites => sprite_index1], tr[:init => :init_objs => obin2], tr[:init => :init_sprites => sprite_index2])

        new_trace_choices[(:init => :init_sprites => sprite_index1 => :mask)] = newsprite.mask
        new_trace_choices[(:init => :init_sprites => sprite_index1 => :color)] = newsprite.color

        #what to do with the second sprite? shift all the others i guess TODO 

        #move all of the objects with those sprites 
        N = tr[:init => :N]

        for obin in 1:N
            # if sprite index 1
            if tr[:init => :init_objs => obin => :sprite_index] == sprite_index1
                pos = tr[:init => :init_objs => obin => :pos]
                newy, newx = pos.y + relativey1, pos.x + relativex1
                new_trace_choices[(:init => :init_objs => obin => :pos)] = Position(newy, newx)
            end 
            #if sprite index 2
            if tr[:init => :init_objs => obin => :sprite_index] == sprite_index2
                pos = tr[:init => :init_objs => obin => :pos]
                newy, newx = pos.y + relativey2, pos.x + relativex2
                new_trace_choices[(:init => :init_objs => obin => :pos)] = Position(newy, newx)
                new_trace_choices[(:init => :init_objs => obin => :sprite_index)] = sprite_index1
            end
        end 


        #for backward choices need new #obin, new_second_sprite_index, vertical_or_horizontal, split_point
        backward_choices[:obin] = obin1
        backward_choices[:exception] = false
        backward_choices[:new_second_sprite_index] = sprite_index2 #aak 
        backward_choices[:vertical_or_horizontal] = 1 #aaaaak 
        backward_choices[:split_point] = 1 #something about displacement 
    end 
    # @show split_merge
    # @show new_trace_choices
    # @show backward_choices

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)    
        
end 


#new merge plan 
#randomly samples 
    
    