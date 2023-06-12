using Gen
using Plots
include("julie_practice_gen.jl")
include("2gamma_sin.jl")

#I think i did this wrong. results are bad 
#or wait actually they are good, they just don't reward being simple and not crazy off the points

 
ys_sine = [2.89, 2.22, -0.612, -0.522, -2.65, -0.133, 2.70, 2.77, 0.425, -2.11, -2.76];

scatter(xs, ys_sine, color="black", label=nothing)

function sine_inference(model, xs, ys_sine, amount_of_computation)
    
    # Create a choice map that maps model addresses (:y, i)
    # to observed values ys[i]. We leave :slope and :intercept
    # unconstrained, because we want them to be inferred.
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys_sine)
        observations[(:y, i)] = y
    end
    
    # Call importance_resampling to obtain a likely trace consistent
    # with our observations.
    (trace, _) = Gen.importance_resampling(model, (xs,), observations, amount_of_computation);
    return trace
end;

traces = [sine_inference(sine_model, xs, ys_sine, 10000) for _=1:10];
gridd(render_trace, traces)

#ok