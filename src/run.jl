include("model.jl")

using Revise
includet("inference.jl"); 
# println(Base.Filesystem.pwd())

#frostbite testing 

#big version
@time traces = particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20,tstart=200, tskip=4)[:,:,:,1:4], 8); #120
#mid version 
#@time traces = particle_filter(3, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8); 
#small version 
#@time traces = particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=170, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8);
#tiny version 
#@time traces = particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1:4], 8);

#first frame testing nvm this is annoying to write

include("html.jl")
include("images.jl")	
#render samples from forward model 
import ..Model: model

# # #generative model only 
# html_gif(new_html(), render_trace(generate(model, (2, 2, 3))[1]), show=true)
# Gen.get_choices(generate(model, (2, 2, 3))[1])





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

