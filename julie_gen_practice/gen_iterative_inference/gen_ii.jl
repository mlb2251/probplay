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
                    println(ys[i])
                end
                println(constraints)
                constraints
            end;


observations = make_constraints(ys); #??make constraints what does it do (defined right above)

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end;

traces    = [first(Gen.importance_resampling(regression_with_outliers, (xs,), observations, 2000)) for i in 1:9]
log_probs = [get_score(t) for t in traces]
println("Average log probability: $(logmeanexp(log_probs))")
Plots.plot([visualize_trace(t) for t in traces]...)

#section 4


# Gen's `generate` function accepts a model, a tuple of arguments to the model,
# and a `ChoiceMap` representing observations (or constraints to satisfy). It returns
# a complete trace consistent with the observations, and an importance weight.  
# In this call, we ignore the weight returned.
(tr, _) = generate(regression_with_outliers, (xs,), observations)#rwo is the func above


#independent Metropolis Hasting, puts all the variables into a single block (to resample)
#slow right???
#each random choice own block is the other extreme




# Perform a single block resimulation update of a trace.
function block_resimulation_update(tr)
    # Block 1: Update the line's parameters
    line_params = select(:noise, :slope, :intercept)
    (tr, _) = mh(tr, line_params)
    
    # Blocks 2-N+1: Update the outlier classifications, each one
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:data => i => :is_outlier))
    end
    
    # Block N+2: Update the prob_outlier parameter
    (tr, _) = mh(tr, select(:prob_outlier))
    
    # Return the updated trace
    tr
end;


#resimulate 500 times 
function block_resimulation_inference(xs, ys, observations)
    observations = make_constraints(ys)
    #initial 
    (tr, _) = generate(regression_with_outliers, (xs,), observations)
    for iter=1:500
        tr = block_resimulation_update(tr)
    end
    tr
end;

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = block_resimulation_inference(xs, ys, observations)
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))


#visualizing  WHOAAA

t, = generate(regression_with_outliers, (xs,), observations)

viz = Plots.@animate for i in 1:500
    global t
    t = block_resimulation_update(t)
    visualize_trace(t; title="Iteration $i/500")
end;
gif(viz)


#part 5

#more general metropolis hastings flavor (tr, did_accept) = mh(tr, custom_proposal, custom_proposal_args)

#proposes new slightly diff slope and intercept 
@gen function line_proposal(current_trace)
    slope ~ normal(current_trace[:slope], 0.5)
    intercept ~ normal(current_trace[:intercept], 0.5)
end;

#uses above function 
(tr, did_accept) = mh(tr, line_proposal, ())#assumes current trace is why empty

function gaussian_drift_update(tr)
    # Gaussian drift on line params
    (tr, _) = mh(tr, line_proposal, ())
    
    # Block resimulation: Update the outlier classifications
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:data => i => :is_outlier))
    end
    
    # Block resimulation: Update the prob_outlier parameter
    (tr, w) = mh(tr, select(:prob_outlier))
    (tr, w) = mh(tr, select(:noise))
    tr
end;

tr1, = generate(regression_with_outliers, (xs,), observations)
tr2 = tr1

viz = Plots.@animate for i in 1:300
    global tr1, tr2
    tr1 = gaussian_drift_update(tr1)
    tr2 = block_resimulation_update(tr2)
    Plots.plot(visualize_trace(tr1; title="Drift Kernel (Iter $i)"), #gaussian drift is better less random
               visualize_trace(tr2; title="Resim Kernel (Iter $i)"))
end;
gif(viz)

#resim got more stuck on a horrestontal line 


#part 6 

