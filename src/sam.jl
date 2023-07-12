using PyCall


struct SAM
    sam::PyCall.PyObject
    mask_generator::PyCall.PyObject
end

global global_sam::Union{Nothing,SAM} = nothing

function sam_init(;device, force=false, checkpoint="$(ENV["HOME"])/proj/shared/sam/sam_vit_h_4b8939.pth", points_per_side=64, points_per_batch=64, kwargs...)
    global global_sam
    SA = pyimport("segment_anything")
    if isnothing(global_sam) || force
        println("Loading SAM from $checkpoint")
        # torch = pyimport("torch")
        sam = SA.sam_model_registry["vit_h"](checkpoint=checkpoint)
        # sam = torch.ao.quantization.quantize_dynamic(sam)
    else
        sam = global_sam.sam
    end
    mask_generator = SA.SamAutomaticMaskGenerator(sam, points_per_side=points_per_side, points_per_batch=points_per_batch, kwargs...)
    global_sam = SAM(sam, mask_generator)

    global_sam.sam.to(device=device)
end

function sam_masks(frames)
    first_frame = UInt8.(round.(frames[:,:,:,1] * 255))
    hwc_frame = permutedims(first_frame, (2,3,1))
    println("Generating masks...")
    @time global_sam.mask_generator.generate(hwc_frame)
end

function sam_clusters(masks)
    masks = sort(masks, by = m -> -m["area"])
    clusters = zeros(size(masks[1]["segmentation"])...)
    for (i,mask) in enumerate(masks)
        clusters[mask["segmentation"]] .= i
    end

    separated = Matrix{Int}[]
    for (i,mask) in enumerate(masks)
        push!(separated, mask["segmentation"] * i)
    end

    clusters, separated
end