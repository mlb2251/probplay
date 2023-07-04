using Atari
using Gen




function full1()
    for _ in 1:8 gen_large() end
    particle_large()
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
    sam_init(device=0)
    # frames = crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20, tstart=200, tskip=4)[:,:,:,1:20]
    frames = load_frames(path)
    masks = sam_masks(frames)
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


function particle_large()
    @time particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20,tstart=200, tskip=4)[:,:,:,1:20], 8);
end

function particle_mid()
    @time particle_filter(3, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8); 
end

function particle_small()
    @time particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=170, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8);
end

function particle_tiny()
    @time particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1:4], 8);
end

function gen_tiny()
    html_body(html_gif(render_trace(generate(model, (2, 2, 3))[1])))
end

function gen_mid()
    html_body(html_gif(render_trace(generate(model, (100, 100, 50))[1])))
end

function gen_large()
    html_body(html_gif(render_trace(generate(model, (210, 160, 100))[1])))
end

function get_choices_tiny()
    Gen.get_choices(generate(model, (2, 2, 3))[1])
end
