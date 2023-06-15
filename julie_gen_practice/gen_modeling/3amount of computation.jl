using Gen
using Plots
include("julie_practice_gen.jl")


trace = do_inference(line_model, xs, ys, 1)#last is amount_of_computation AWFUL
render_trace(trace)


trace = do_inference(line_model, xs, ys, 10)#last is amount_of_computation OK
render_trace(trace)


trace = do_inference(line_model, xs, ys, 100)#last is amount_of_computation OK
render_trace(trace)

trace = do_inference(line_model, xs, ys, 1000)#last is amount_of_computation GOOD, 1 sec
render_trace(trace)

trace = do_inference(line_model, xs, ys, 10000)#last is amount_of_computation GOOD, 1 sec
render_trace(trace)