using Gen
using Distributions
#using Random

#HELPERS

function scores_from_heatmap(heatmap, other_hiwi_func=nothing)
    """gets scores from heatmap,
    optional second arg adds another weight into the score (ex: distance from a point)"""
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

function position_from_flatmatrixindex(index, H, W)
    """converts into position from an index of a 1d flattened matrix"""
    x = floor(index / H) + 1
    y = index % H

    return Position(y, x)
end

function flatmatrixindex_from_position(pos, H, W)
    """converts into 1d flattened matrix index from a position"""
    return (pos.x-1)*H + pos.y
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
    """
    edits trace to shift all sprite indicies down or up (depending on shift_func)
    """
    function shiftonething(thing, i)
        new_trace_choices[(:init => :init_sprites => shift_func(i) => thing)] = tr[:init => :init_sprites => i => thing] 
    end 

    for i in min_sprite:max_sprite
        shiftonething(:mask, i)
        shiftonething(:color, i)
        shiftonething(:height, i)
        shiftonething(:width, i)

    end 
end 

#SPRITE STUFFFF-----------------------------------------------------------------------------------------------
@gen function add_remove_sprite_random(tr)
"""
non data driven version 
if remove samples sprite id
if add samples sprite id, sprite mask, sprite color, list of obj ids, list of positions"""
    num_sprite_types = tr[:init => :num_sprite_types]
    if({:add_or_remove} ~ bernoulli(num_sprite_types == 1 ? 1 : 0.5)) 
        #adding 
        C, H, W = size(tr[:init => :observed_image])
        height ~ uniform_discrete(1,H)
        width ~ uniform_discrete(1,W)
        mask ~ bernoulli_2d(0.8, height, width) #change this
        color ~ rgb_dist()
        
        
        N = tr[:init => :N]
        num_objs ~ poisson_plus_1(0.5)

        #TODO make them not all add to the end for perfect reversability 
        #sample position for each new object
        positions ~ allpositions(collect(1:num_objs), [H for _ in 1:num_objs], [W for _ in 1:num_objs])
    else 
        #removing 
        rm_sprite_index ~ uniform_discrete(1, num_sprite_types)
    end 
end 


function update_min_max(pos, miny, minx, maxy, maxx)
    """helper for get_new_ff_sprite in cropping the sprite"""
    if pos.y < miny
        miny = pos.y
    end
    if pos.y > maxy
        maxy = pos.y
    end
    if pos.x < minx
        minx = pos.x
    end
    if pos.x > maxx
        maxx = pos.x
    end
    return miny, minx, maxy, maxx
end


@gen function get_color_bernoulli(i, observed_image, pos, base_color)
    """samples a bernoulli based on how close the color at pos is to base_color"""
    color_diff = sum(abs.(observed_image[:, pos.y, pos.x] - base_color))/3 #ranges from 0 to 1	
    in_mask ~ bernoulli(1 - color_diff) #if near color, sampling this for full support(will use to decide to ask to mask and neibors to queue)
end

get_all_color_bernoulli = Map(get_color_bernoulli)


@gen function get_new_ff_sprite(tr, heatmap)
    """makes a new sprite using a floodfill + bernoulli"""
    #samples a place to start filling from based on heatmap
    H, W = size(heatmap)
    scores = scores_from_heatmap(heatmap)
    flat_matrix_place_index ~ categorical(scores)
    base_pos = position_from_flatmatrixindex(flat_matrix_place_index, H, W) 

    #get the color of the place
    observed_image = tr[:init => :observed_image]
    base_color = observed_image[:, base_pos.y, base_pos.x]

    #floodfill to get mask of a sprite of close to that color 
    massive_mask = fill(0, (H,W))
    pos_queue = [base_pos]
    finished_queue = []

    miny, minx = H, W
    maxy, maxx = 1, 1

    
    while length(pos_queue) > 0
        #increasingnumber += 1
        # @show pos_queue
        # @show finished_queue
        pos = popfirst!(pos_queue)
        push!(finished_queue, pos)
        #color difference 
        color_diff = sum(abs.(observed_image[:, pos.y, pos.x] - base_color))/3 #ranges from 0 to 1, make not have 0 or 100 
        #if near color, sampling this for full support, add to mask and add neibors to queue
        distance = ((pos.y-base_pos.y)^2 + (pos.x-base_pos.x)^2)^0.5
        in_mask = {(:in_mask,pos.y,pos.x)} ~ bernoulli(((1 - color_diff)*0.99+0.005)^(distance))#fix 
        #maybe include smth abt distance to base_pos 
        if in_mask 
            miny, minx, maxy, maxx = update_min_max(pos, miny, minx, maxy, maxx)
            #add to mask
            massive_mask[pos.y, pos.x] = 1
            #add neighbors to pos_queue
            neighbor_list = ((0,-1), (-1,0), (0,1), (1,0), (1,1), (1,-1), (-1,1), (-1,-1))
            for dydx in neighbor_list
                dy, dx = dydx
                #if in image bounds 
                if pos.y + dy > 0 && pos.y + dy <= H && pos.x + dx > 0 && pos.x + dx <= W
                    neighbor_pos = Position(pos.y + dy, pos.x + dx)
                    if !((neighbor_pos in finished_queue) || (neighbor_pos in pos_queue))
                        push!(pos_queue, neighbor_pos)
                    end
                end
            end 
        end 
    end 

    # #filling in the rest false #check how legit this is OH shoot i forgot about the position distinction 
    # for i in 1:H
    #     for j in 1:W
    #         if !(Position(i, j) in finished_queue)
    #             in_mask = {(:in_mask,pos.y,pos.x)} ~ bernoulli(0.0001)
    #             if in_mask
    #                 miny, minx, maxy, maxx = update_min_max(pos, miny, minx, maxy, maxx)
    #                 massive_mask[pos.y, pos.x] = 1
    #             end
    #         end
    #     end 
    # end

    #shrink the mask to the smallest possible and get the position of massive_mask 
    mask_pos = Position(miny, minx)
    mask = massive_mask[miny:maxy, minx:maxx]
    #rendering for testing 
    html_body(render_matrix(mask, 3))
    
    r ~ beta(get_alpha(base_color[1]), 1)
    g ~ beta(get_alpha(base_color[2]), 1)
    b ~ beta(get_alpha(base_color[3]), 1)

    color = [r, g, b]
    sprite = Sprite(mask, color)
    #return sprite, mask_pos
    return sprite 
end 



@gen function dd_add_remove_sprite_random(tr, heatmap)
    """
    data driven 
    if remove samples sprite id
    if add samples sprite id, sprite mask, sprite color, list of obj ids, list of positions"""
    num_sprite_types = tr[:init => :num_sprite_types]
    if({:add_or_remove} ~ bernoulli(num_sprite_types == 1 ? 1 : 0.5)) 
        #adding 
        #make this data driven 
        C, H, W = size(tr[:init => :observed_image])
        #sprite, mask_pos ~ get_new_ff_sprite(tr, heatmap) #how to do this 
        #@show get_new_ff_sprite(tr, heatmap)

        #@show get_choices(get_new_ff_sprite(tr, heatmap))
        #sprite = get_new_ff_sprite(tr, heatmap) #new supmap of choicemap called sprite with the choices

        #@show generate(get_new_ff_sprite, (tr, heatmap))
        
        #map[:sprite] = get_new_ff_sprite(tr, heatmap)
        #sprite = {:sprite} ~ get_new_ff_sprite(tr, heatmap) #new supmap of choicemap called sprite with the choices 
        
        sprite ~ get_new_ff_sprite(tr, heatmap)

        num_objs ~ poisson_plus_1(0.5)

        #todo make them not all add to the end 
        #sample position for each new object, to do make data drivennnn
        #todo incorporate mask_pos here!! to make more dd
        positions ~ allpositions(collect(1:num_objs), [H for _ in 1:num_objs], [W for _ in 1:num_objs])
        #@show positions 
        return sprite 
    else 
        #removing 
        #todo make data driven 
        rm_sprite_index ~ uniform_discrete(1, num_sprite_types)
    end 
end 



function add_remove_sprite_involution(tr, add_remove_random, forward_retval, proposal_args)
    """non dd, involution adds sprite and some objects of it or removes the sprite and all objects of it"""
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


function dd_add_remove_sprite_involution(tr, add_remove_random, forward_retval, proposal_args)
    @show add_remove_random
    #@show tr
    new_trace_choices = choicemap()
    backward_choices = choicemap()

    if add_remove_random[:add_or_remove]
        #add the new sprite , when we add the ability to add a middle sprite, make sure to deal with the cascade
        sprite = forward_retval 
        snp1num = tr[:init => :num_sprite_types] + 1
        new_trace_choices[:init => :num_sprite_types] = snp1num

        #@show sprite
        new_trace_choices[:init => :init_sprites => snp1num => :mask] = sprite.mask
        new_trace_choices[:init => :init_sprites => snp1num => :color] = sprite.color
        #UHH BE CAREFUL WE DON"T NEED THESE 
        # new_trace_choices[:init => :init_sprites => snp1num => :height] = sprite.height
        # new_trace_choices[:init => :init_sprites => snp1num => :width] = sprite.width

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

        
        #remove each object
        #remember to change N 
        # newN = tr[:init => :N]
        Nremoved = 0
        oneposTEMP = Position(1,1)	

        #remove all objects of that sprite type, shift all the other objects down their ois and if higher si down one
        old_sindicies = [] 
        change_plan = [] #1 means removed, negative/zero means how much to go down
        for i in 1:tr[:init => :N] 
            si = tr[:init => :init_objs => i => :sprite_index]
            #push!(old_sindicies, si)
            if si == sprite_index
                #push!(change_plan, 1)
                pos = tr[:init => :init_objs => i => :pos]
                backward_choices[:positions => i => :pos] = pos
                oneposTEMP = pos 
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


        #SOMETHING LIKE THIS??
        #backward_choices[:sprite] = Sprite(old_mask, tr[:init => :init_sprites => sprite_index => :color]) #mayyybe
        old_mask_vol = size(old_mask)[1]*size(old_mask)[2]
        for i in 1:old_mask_vol
            backward_choices[:sprite => (:in_mask,i)] = true
        end 
        backward_choices[:sprite => :r] = tr[:init => :init_sprites => sprite_index => :color][1]
        backward_choices[:sprite => :g] = tr[:init => :init_sprites => sprite_index => :color][2]
        backward_choices[:sprite => :b] = tr[:init => :init_sprites => sprite_index => :color][3]
        #next line tech incorrect 
        @show oneposTEMP
        backward_choices[:sprite => :flat_matrix_place_index] = flatmatrixindex_from_position(oneposTEMP, size(tr[:init => :observed_image])[1], size(tr[:init => :observed_image])[2])
        #how do I set backward retval 

        
        #set_submap(backward_choices, :sprite, get_submap(add_remove_random[:sprite]))
        #need info like ff about how this sprite was made 
        #LOOK HERE 

        #backward_choices[:sprite] = forward_retval#this is so beyond incorrect 
        backward_choices[:num_objs] = Nremoved
        #backward_choices[:positions] = positions
    end
    #@show new_trace_choices
    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    #@show new_trace
    (new_trace, backward_choices, weight)
end 

#SIZE STUFF
@gen function get_random_size(tr, i)
    #todo make dd

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
    "grows or shrinks sprite "
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


@gen function get_random_hi_wi(tr, i)
    """samples random pixel index in mask (to switch on off)"""
    mask = tr[:init => :init_sprites => i => :mask]
    height, width = size(mask)
    hi ~ uniform_discrete(1, height)	
    wi ~ uniform_discrete(1, width)
end 


function mask_involution(tr, hi_wi, forward_retval, proposal_args)#what are last two for, prop[1] is i??
    """swaps pixel on of off at an index in a sprite mask """
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

@gen function dd_get_random_new_color(tr, i)
    """proposes a new color for a sprite based on the average observed color of that type sprite
    slow, probably removing"""
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

    #samples near the color 
    rnew ~ beta_with_peak(ravg)
    gnew ~ beta_with_peak(gavg)
    bnew ~ beta_with_peak(bavg)
end 



function color_involution(tr, colors, forward_retval, proposal_args)
    """changes the color of a sprite"""
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
    #backward_choices[:flatmatrixindex] = (pos.x-1)*H + pos.y
    backward_choices[:flatmatrixindex] = flatmatrixindex_from_position(pos, H, W)


    flatmatrixindex = newspot[:flatmatrixindex]
    # newx = floor(flatmatrixindex / H) + 1
    # newy = flatmatrixindex % H

    newpos = position_from_flatmatrixindex(flatmatrixindex, H, W)
    new_trace_choices[(:init => :init_objs => i => :pos)] = newpos

    new_trace, weight, = update(tr, get_args(tr), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end 



#ADD REMOVE OBKECT 
@gen function get_add_remove_object(tr)
    """randomly chooses an object to remove or add
    todo make data driven """
    n = tr[:init => :N]
    #@show n 
    num_sprite_types = tr[:init => :num_sprite_types]
    H, W = size(tr[:init => :observed_image])

    if({:add_or_remove} ~ bernoulli(n == 1 ? 1 : 0.1))
        #adding an object 
        ypos ~ uniform_discrete(1, H)
        xpos ~ uniform_discrete(1, W)
        
        sprite_index ~ uniform_discrete(1, num_sprite_types)
        add_obj_index ~ uniform_discrete(1, n+1)


    else 
        #remove object sample index 
        remove_obj_index ~ uniform_discrete(1, n)
        
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

        # #FOR DEBUGGING 
        # spriteind = tr[:init => :init_objs => remove_obj_index => :sprite_index]
        # mask = tr[:init => :init_sprites => spriteind => :mask]
        # render_matrix(mask)

       
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
    
