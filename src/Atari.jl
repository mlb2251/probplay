module Atari
export html_gif, html_img, html_new, render_trace, model, particle_filter, load_frames, crop, html_fresh, html_render, html_body, fresh, render, sam_masks, sam_init, sam_clusters, color_labels, html_table, Object, Sprite, draw, draw_region, Vec, set_pos, set_mask, set_sprite, set_color
# include("gen_mods.jl")
include("model.jl")
include("html.jl")
include("sam.jl")
include("inference.jl");

end # module Atari
