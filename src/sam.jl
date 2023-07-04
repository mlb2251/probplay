using PyCall


struct SAM
    sam::PyCall.PyObject
    mask_generator::PyCall.PyObject
end

global global_sam::Union{Nothing,SAM} = nothing

function sam_init(;device)
    global global_sam
    if isnothing(global_sam)
        println("initializing SAM")
        SA = pyimport("segment_anything")
        # torch = pyimport("torch")
        sam = SA.sam_model_registry["vit_h"](checkpoint="../shared/sam/sam_vit_h_4b8939.pth")
        # sam = torch.ao.quantization.quantize_dynamic(sam)
        mask_generator = SA.SamAutomaticMaskGenerator(sam)
        global_sam = SAM(sam, mask_generator)
    end

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
        # @assert Int.(clusters[mask["segmentation"]]) == 0
        clusters[mask["segmentation"]] .= i
    end

    # separated = cat(map((i,m) -> m["segmentation"]*i, enumerate(masks))...,dims=2)
    separated = Matrix{Int}[]
    for (i,mask) in enumerate(masks)
        # @assert Int.(clusters[mask["segmentation"]]) == 0
        push!(separated, mask["segmentation"] * i)
        push!(separated, fill(1,size(mask["segmentation"],1),5))
    end


    clusters, cat(separated...,dims=2)
end