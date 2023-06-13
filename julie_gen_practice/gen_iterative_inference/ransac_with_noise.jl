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

@gen function ransac_proposal_noise_based(prev_trace, xs, ys)
    params = RANSACParams(10, 3, 1.)
    (slope_guess, intercept_guess) = ransac(xs, ys, params)
    slope ~ normal(slope_guess, 0.1)
    intercept ~ normal(intercept_guess, 1.0)
    return params, slope, intercept # (return values just for testing)
end;