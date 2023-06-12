using Gen
using Plots
#import julie_practice_gen.jl
include("julie_practice_gen.jl")


function render_trace(trace; show_data=true)
    
    # Pull out xs from the trace
    xs, = get_args(trace)
    
    xmin = minimum(xs)
    xmax = maximum(xs)

    # Pull out the return value, useful for plotting
    y = get_retval(trace)
    
    # Draw the line
    test_xs = collect(range(-5, stop=5, length=1000))
    fig = plot(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                xlim=(xmin, xmax), ylim=(xmin, xmax))

    if show_data
        ys = [trace[(:y, i)] for i=1:length(xs)]
        
        # Plot the data set
        scatter!(xs, ys, c="black", label=nothing)
    end
    
    return fig
end;


function grid(renderer::Function, traces)
    Plots.plot(map(renderer, traces)...)
end;

@gen function sine_model(xs::Vector{Float64})
    
    # < your code here, for sampling a phase, period, and amplitude >
    amplitude = ({:amp} ~ gamma(1, 1))
    period = ({:per} ~ gamma(1, 1))
    phase= ({:pha} ~ uniform(0, 2*pi))

    print("did we make it here")
    function y(x)
        return 1 # amplitude*sin(2*pi*x/(0.0000000001+ period)+phase) 
    end
    
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(y(x), 0.1)
    end
    
    return y # We return the y function so it can be used for plotting, below. 
end;


traces = [Gen.simulate(sine_model, (xs,)) for _=1:12];

gridd(render_trace, traces)