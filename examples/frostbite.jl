using Atari
using Gen




function full1()
    for _ in 1:8 gen_large() end
    particle_large()
end

function sam()
    sam_init(device=0)
    # frames = crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20, tstart=200, tskip=4)[:,:,:,1:20]
    frames = crop(load_frames("atari-benchmarks/frostbite_1"), tstart=200, tskip=4)[:,:,:,1:20]
    masks = sam_masks(frames)
    clusters,separated = Atari.sam_clusters(masks)

    html_body(
        "<h3>Observation</h3>",
        html_img(frames[:,:,:,1]),
        "<h3>Segmentation</h3>",
        html_img(color_labels(clusters)),
        "<h3>Individual Segments</h3>",
        html_img(color_labels(separated), width="$(100*length(masks))px")
    )
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
