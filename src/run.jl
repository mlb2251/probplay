include("model.jl")
include("inference.jl")

using Revise
includet("inference.jl"); 
println(Base.Filesystem.pwd())
@time traces = particle_filter(6, crop(load_frames("atari-benchmarks/frostbite_1"), top=120, bottom=25, left=20, tskip=4)[:,:,:,1:4], 8); #termal relative to proj directory, but includes are 


