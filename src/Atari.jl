module Atari
export html_gif, html_new, render_trace, model, particle_filter, load_frames, crop, html_fresh, html_render, html_body, fresh, render

include("model.jl")
include("html.jl")
include("inference.jl"); 
end # module Atari
