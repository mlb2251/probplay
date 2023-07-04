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
        clusters[mask["segmentation"]] .= i
    end

    separated = Matrix{Int}[]
    for (i,mask) in enumerate(masks)
        push!(separated, mask["segmentation"] * i)
    end

    clusters, separated
end