using Atari
using Gen


square!(sprites, color, side) = rect!(sprites, color, side, side)
function rect!(sprites, color, h, w)
    push!(sprites, Sprite(ones(Bool, h, w), color))
    length(sprites)
end
function circle!(sprites, color, radius)
    side = 2*radius + 1
    mask = zeros(Bool, side, side)
    for y in 1:side
        for x in 1:side
            if (y - radius - 1)^2 + (x - radius - 1)^2 < radius^2 + 1.5 # 1.5 just makes it look a little smoother
                mask[y,x] = true
            end
        end
    end
    push!(sprites,Sprite(mask, color))
    length(sprites)
end
function tff(;particles=3, steps=200, step_chunk=50, which=:big)
    #@show particles, steps, step_chunk
    fresh(); 
    first_frame(particles=particles, steps=steps, step_chunk=step_chunk, which=which)
    
    render();
end

function shape_test()
    objs = Object[]
    sprites = Sprite[]
    c = circle!(sprites,[.7,0.,0.], 3)
    r1 = rect!(sprites,[0.,0.,.7], 3, 4)
    s = square!(sprites, [0.,.7,0.], 5)
    objs = [Object(c, Vec(5,15)), Object(r1, Vec(4,5)), Object(s, Vec(10,5))]
    obs = draw(20, 40, objs, sprites)
    html_body(html_img(obs, width="400px"))
end

function code_test()
    H,W = 40,40
    T = 20
    objs = Object[]
    sprites = Sprite[]
    c = circle!(sprites,[.7,0.,0.], 3)

    env = new_env();
    env.code_library = [
        CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (normal_vec (get_attr (get_local 1) pos) 0.3) (get_attr (get_local 1) 1) 0.5))")),
        CFunc(parse(SExpr,"(set_attr (get_local 1) pos (normal_vec (get_attr (get_local 1) pos) 1.0))"))
       # CFunc(parse(SExpr,"(set_attr (get_local 1) pos (+ (normal_vec (get_attr (get_local 1) pos) 0.3) (get_attr (get_local 1) 1) 0.5))")),
    ]
    # push!(env.code_library, CFunc(parse(SExpr,
    #     "(set_attr (get_local 1) pos (normal_vec (get_attr (get_local 1) 1) 1.0))"
    # )))
        # CSetAttr(CGetLocal(1), 0, CNormalVec(CGetAttr(CGetLocal(1), 0), CFloat(1.0)))))

    objs = [Object(c, Vec(20,20), [Vec(1.0,1.0)])]
    first_frame = draw(H, W, objs, sprites)

    # vel = Vec(1,2)

    frames = zeros(Float64, 3, H, W, T)
    frames[:,:,:,1] = first_frame


    func = 1

    @time for t in 2:T
        call_func(env.code_library[func], Any[objs[1]], env)
        frames[:,:,:,t] = draw(H, W, objs, sprites)
        html_body(html_img(frames[:,:,:,t], width="400px"))
        if t == 10
            func = 2
        end
    end
    html_body("<code> (set_attr (get_local 1) pos (normal_vec (get_attr (get_local 1) pos) 1.0)) </code> <br><br>")
    html_body(html_gif(frames, width="400px"))
end


function bouncing_ball()
    H,W = 40,40
    T = 20
    sprites = Sprite[]
    c = circle!(sprites,[.7,0.,0.], 3)
    objs = [Object(c, Vec(5,15))]
    first_frame = draw(H, W, objs, sprites)

    vel = Vec(1,2)

    frames = zeros(Float64, 3, H, W, T)
    frames[:,:,:,1] = first_frame

    for t in 2:T
        objs[1] = set_pos(objs[1], objs[1].pos + vel)
        frames[:,:,:,t] = draw(H, W, objs, sprites)

        # vel += Vec(rand()-.7, rand()-.7)
    end

    html_body(html_gif(frames, width="400px"))
end



function full1()
    html_body("<h2>Samples from the prior</h2>")
    for _ in 1:8 gen_large() end
    particle_large_birds()
end

function segment_table(masks, mask_imgs)
    table = fill("", 8, length(masks)+1)
    keys = ["predicted_iou", "area", "stability_score", "bbox", "point_coords", "crop_box"]
    table[1,1] = "Segmentation"
    table[2,1] = "Mask ID"
    for (k,key) in enumerate(keys)
        table[k+2,1] = key
    end

    for i in eachindex(masks)
        table[1,i+1] = html_img(mask_imgs[i]);
        table[2,i+1] = "Mask $i"
        for (k,key) in enumerate(keys)
            value = masks[i][key]
            if key == "predicted_iou" || key == "stability_score"
                value = round(value,sigdigits=6)
            end
            table[k+2,i+1] = "$value"
        end
    end
    html_table(table)
end

function sam(path="atari-benchmarks/frostbite_1")
    @time sam_init(device="cpu", points_per_side=1, points_per_batch=1)
    # frames = crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20, tstart=200, tskip=4)[:,:,:,1:20]
    frames = load_frames(path)
    @time masks = sam_masks(frames)
    clusters, separated = Atari.sam_clusters(masks)
    mask_imgs = color_labels(separated...)

    html_body(
        "<h3>Frostbite</h3>",
        html_table(["Observation" "Segmentation"; html_img(frames[:,:,:,1]) html_img(color_labels(clusters)[1])]),
        "<h3>Segments</h3>",
        segment_table(masks,mask_imgs)
    )

end

function sam_everything()
    sam_init(device=0)
    for (i,game_path) in enumerate(filter(x -> occursin("-v5",x), readdir("atari-benchmarks/variety",join=true)))
        @show i
        frames = load_frames(game_path)
        masks = sam_masks(frames)
        clusters, separated = Atari.sam_clusters(masks)
        mask_imgs = color_labels(separated...)

        html_body(
            "<h2>$game_path</h2>",
            html_table(["Observation" "Segmentation"; html_img(frames[:,:,:,1]) html_img(color_labels(clusters)[1])]),
            "<h3>Segments</h3>",
            segment_table(masks,mask_imgs)
        )
    end
end

function particle_large_birds(T=20)
    @time particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20,tstart=200, tskip=4)[:,:,:,1:T], 8);
end

function particle_full_frostbite()
    @time particle_filter(3, crop(load_frames("atari-benchmarks/frostbite_1"), tskip=4)[:,:,:,1:30], 8);
end

function particle_large()
    @time particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20,tstart=1, tskip=4)[:,:,:,1:30], 8);
end

# particle_large_birds();

function particle_mid()
    @time particle_filter(3, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8); 
end

function particle_small()
    @time particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=170, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8);
    print("TESTjfkaldsjakljfkla;dsjfkl;sjfkl;s")
end

function particle_tiny(;T=4, particles=1, mh_steps_init=100, mh_steps=50)
    @time particle_filter(particles, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=20)[:,:,:,1:T], 2; mh_steps_init=mh_steps_init, mh_steps=mh_steps);
end

function gen_tiny()
    html_body(html_gif(render_trace(generate(model, (2, 2, 3))[1])))
end

function gen_mid()
    html_body(html_gif(render_trace(generate(model, (100, 100, 50))[1])))
end

function gen_large()
    tr,wt = generate(model, (210, 160, 100))
    @assert !isnan(wt)
    html_body(html_gif(render_trace(tr)))
end

function get_choices_clean(tr)
    choices = Gen.get_choices(tr)
    # @show typeof(tr)
    # @show typeof(choices)
    choices
end

function get_choices_tiny()
    Gen.get_choices(generate(model, (2, 2, 3))[1])
end

function first_frame(;particles=8, steps=1000, step_chunk=50, which=:big)
    #@show particles, steps, step_chunk
    #first_frame = crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=25, left=20, tskip=4)[:,:,:,1]	
    #@time particle_filter(1, crop(load_frames("atari-benchmarks/frostbite_1"), top=100, bottom=20, left=20, tskip=4)[:,:,:,1:4], 4)
    # @time particle_filter(1, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1:4], 4)

    #small
    #frame = crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1]

    #tiny
    frame = if which === :big
        crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20,tstart=1, tskip=4)[:,:,:,1]
    elseif which === :tiny
        crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1]
    end


    (C,H,W) = size(frame)

    html_body("<h1>Observations</h1>", html_img(frame))
    
    Atari.process_first_frame_v2(frame; num_particles=particles, steps=steps, step_chunk=step_chunk)
end


function test_involution()
    frame = crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1]


    (C,H,W) = size(frame)

    html_body("<h1>Observations</h1>", html_img(frame))
    Atari.test_one_involution(frame)
end 


function particle_positions(;T=4, particles=1, mh_steps_init=100, mh_steps=50)
    @time position_particle_filter(particles, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=20)[:,:,:,1:T], 2; mh_steps_init=mh_steps_init, mh_steps=mh_steps);
end