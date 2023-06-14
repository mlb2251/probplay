import Random, Logging
using Gen, Plots
import StatsBase
include("regression_viz_import.jl")
include("gen_ii.jl")

# Disable logging, because @animate is verbose otherwise
Logging.disable_logging(Logging.Info);


# Modify the function below (which currently is just a copy of `ransac_proposal`) 
# as described above so that implements a RANSAC proposal with inlier 
# status decided by the noise parameter of the previous trace
# (do not modify the return value, which is unneccessary for a proposal, 
# but used for testing)

#colons are like strings. called symbols 


@gen function ransac_proposal_noise_based(prev_trace, xs, ys)
    #print("PT", prev_trace)
    noise = prev_trace[:noise]
    #println("N", noise)
    params = RANSACParams(10, 3, noise)
    (slope_guess, intercept_guess) = ransac(xs, ys, params)
    slope ~ normal(slope_guess, 0.1)
    intercept ~ normal(intercept_guess, 1.0)

    return params, slope, intercept # (return values just for testing)
end;



function ransac_update_noise_based(tr)
    # Use RANSAC to (potentially) jump to a better line
    (tr, _) = mh(tr, ransac_proposal_noise_based, (xs, ys))
    # Refining the parameters
    for j=1:20
        (tr, _) = mh(tr, select(:prob_outlier))
        (tr, _) = mh(tr, select(:noise))
        (tr, _) = mh(tr, line_proposal, ())
        # Reclassify outliers
        for i=1:length(get_args(tr)[1])
            (tr, _) = mh(tr, select(:data => i => :is_outlier))
        end
    end
    tr
end;
function ransac_inference_noise_based(xs, ys, observations)
    # Use an initial epsilon value of 1.
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    slope_intercept_init = choicemap()
    slope_intercept_init[:slope] = slope
    slope_intercept_init[:intercept] = intercept
    (tr, _) = generate(regression_with_outliers, (xs,), merge(observations, slope_intercept_init))
    for iter=1:5
        tr = ransac_update_noise_based(tr)
    end
    tr
end

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = ransac_inference_noise_based(xs, ys, observations)
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))

