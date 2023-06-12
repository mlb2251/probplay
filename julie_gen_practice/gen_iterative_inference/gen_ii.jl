import Random, Logging
using Gen, Plots

include("regression_viz_import.jl")

# Disable logging, because @animate is verbose otherwise
Logging.disable_logging(Logging.Info);

 #stuff from https://www.gen.dev/tutorials/iterative-inference/tutorial

#!!this more complicated syntax still needed in loops!!
# x = ({:x} ~ normal(0, 1))
# slope = ({:slope} ~ normal(0, 1))

#  # Desugars to "x = {:x} ~ normal(0, 1)"
# x ~ normal(0, 1)
# # Desugars to "slope = {:slope} ~ normal(0, 1)"
# slope ~ normal(0, 1)

@gen function regression_with_outliers(xs::Vector{<:Real})
    # First, generate some parameters of the model. We make these
    # random choices, because later, we will want to infer them
    # from data. The distributions we use here express our assumptions
    # about the parameters: we think the slope and intercept won't be
    # too far from 0; that the noise is relatively small; and that
    # the proportion of the dataset that don't fit a linear relationship
    # (outliers) could be anything between 0 and 1.
    slope ~ normal(0, 2)
    intercept ~ normal(0, 2)
    noise ~ gamma(1, 1)
    prob_outlier ~ uniform(0, 1)
    
    # Next, we generate the actual y coordinates.
    n = length(xs)
    ys = Float64[]
    
    for i = 1:n
        # Decide whether this point is an outlier, and set
        # mean and standard deviation accordingly
        if ({:data => i => :is_outlier} ~ bernoulli(prob_outlier))
            (mu, std) = (0., 10.)
        else
            (mu, std) = (xs[i] * slope + intercept, noise)
        end
        # Sample a y value for this point
        push!(ys, {:data => i => :y} ~ normal(mu, std))
    end
    ys
end;


# Generate nine traces and visualize them

xs     = collect(range(-5, stop=5, length=20))
traces = [Gen.simulate(regression_with_outliers, (xs,)) for i in 1:9];
Plots.plot([visualize_trace(t) for t in traces]...)

print("yay")



function make_synthetic_dataset(n)
    Random.seed!(1)
    prob_outlier = 0.2
    true_inlier_noise = 0.5
    true_outlier_noise = 5.0
    true_slope = -1
    true_intercept = 2
    xs = collect(range(-5, stop=5, length=n))
    ys = Float64[]
    for (i, x) in enumerate(xs)
        if rand() < prob_outlier
            y = randn() * true_outlier_noise
        else
            y = true_slope * x + true_intercept + randn() * true_inlier_noise
        end
        push!(ys, y)
    end
    (xs, ys)
end
    
(xs, ys) = make_synthetic_dataset(20);
Plots.scatter(xs, ys, color="black", xlabel="X", ylabel="Y", 
              label=nothing, title="Observations - regular data and outliers")

              function make_constraints(ys::Vector{Float64})
                constraints = Gen.choicemap()
                for i=1:length(ys)
                    constraints[:data => i => :y] = ys[i]
                end
                constraints
            end;


observations = make_constraints(ys);

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end;

traces    = [first(Gen.importance_resampling(regression_with_outliers, (xs,), observations, 2000)) for i in 1:9]
log_probs = [get_score(t) for t in traces]
println("Average log probability: $(logmeanexp(log_probs))")
Plots.plot([visualize_trace(t) for t in traces]...)

#finished section 3