include("model.jl")
#include("inference.jl")

# using Revise
# includet("inference.jl"); 
# println(Base.Filesystem.pwd())

#frostbite testing 
@time traces = particle_filter(6, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8); #termal relative to proj directory, but includes are 


include("html.jl")
include("images.jl")	
#render samples from forward model 
import ..Model: model

#generative model only 
html_gif(new_html(), render_trace(generate(model, (200, 200, 5))[1]), show=true);




##small tests 
# trace = generate(model, (10, 5, 5))[1]
# args = get_args(trace)
# #@show trace

# testy = trace[:init => :N]
# @show testy

# testing = objs_from_trace(trace,2)#t = 2 for testing 
# # @show testing
# trace = generate(model, (10, 20, 5))[1]

# (H,W,T) = get_args(trace)

# #@show T

# stack([render_trace_frame(trace, t) for t in 0:T-1])

