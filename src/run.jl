include("model.jl")
#include("inference.jl")

# using Revise
# includet("inference.jl"); 
# println(Base.Filesystem.pwd())
# @time traces = particle_filter(6, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8); #termal relative to proj directory, but includes are 


include("html.jl")
include("images.jl")	
#render samples from forward model 
import ..Model: model
html_gif(new_html(), render_trace(generate(model, (100, 200, 5))[1]), show=true);

# trace = generate(model, (10, 20, 5))[1]

# (H,W,T) = get_args(trace)

# #@show T

# stack([render_trace_frame(trace, t) for t in 0:T-1])

