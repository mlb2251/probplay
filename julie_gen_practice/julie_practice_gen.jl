using Gen
using Plots
#import RecipesBase: grid


# x = {:initial_x} ~ normal(0, 1)
# if x < 0
#     x = x + ({:addition_to_x} ~ normal(2, 1))
# print("hi")
# print(x)
# end
# print("hi")
# print(x)

@gen function line_model(xs::Vector{Float64})
    # We begin by sampling a slope and intercept for the line.
    # Before we have seen the data, we don't know the values of
    # these parameters, so we treat them as random choices. The
    # distributions they are drawn from represent our prior beliefs
    # about the parameters: in this case, that neither the slope nor the
    # intercept will be more than a couple points away from 0.
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    
    # We define a function to compute y for a given x
    function y(x)
        return slope * x + intercept
    end

    # Given the slope and intercept, we can sample y coordinates
    # for each of the x coordinates in our input vector.
    for (i, x) in enumerate(xs)
        # Note that we name each random choice in this loop
        # slightly differently: the first time through,
        # the name (:y, 1) will be used, then (:y, 2) for
        # the second point, and so on.
        ({(:y, i)} ~ normal(y(x), 0.1))
    end

    # Most of the time, we don't care about the return
    # value of a model, only the random choices it makes.
    # It can sometimems be useful to return something
    # meaningful, however; here, we return the function `y`.
    return y
end;

xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];

y = line_model(xs)
println("TESTSSSS")
println(y(0), y(1), y(2))

#y (generic function with 1 method)
trace = Gen.simulate(line_model, (xs,));
#println(trace)
println("slope random choice: ", trace[:slope])
println("return value: ", Gen.get_retval(trace), trace[]) #same thing


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

render_trace(trace)


function gridd(renderer::Function, traces)
    Plots.plot(map(renderer, traces)...)
end;

traces = [Gen.simulate(line_model, (xs,)) for _=1:12]
gridd(render_trace, traces)

