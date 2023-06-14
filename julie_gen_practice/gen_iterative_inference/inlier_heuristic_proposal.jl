import Random, Logging
using Gen, Plots
import StatsBase
include("regression_viz_import.jl")
include("gen_ii.jl")

# Disable logging, because @animate is verbose otherwise
Logging.disable_logging(Logging.Info);


@gen function inlier_heuristic_proposal(prev_trace, xs, ys)
    # Put your code below, ensure that you compute values for
    # inlier_slope, inlier_intercept and delete the two placeholders
    # below.


    inlier_slope, inlier_intercept = ransac(xs, ys, RANSACParams(10, 3, 1.))


    # print(typeof(xs))
    # inlier_slope, inlier_intercept = 100, 100 #DELETE LMAO
    # xs_inliers(), ys_inliers() = A(Float64[],[]), A(Float64[],[])


    # ypred = intercept .+ slope * xs 
    # inliers = abs.(ys - ypred) .< params.eps #just one or zero 

    # for yindex in range(length(ys))
    #     if inliers[yindex] == 1
    #         append!(ys_inliners(), ys[yindex])

    #     end
    # end 

    # print(ys_inliers)

    #get only the xs and ys within epsilon 
    # xs_inliers = ypred = intercept .+ slope * xs

        # # count the number of inliers for this (slope, intercept) hypothesis
        # inliers = abs.(ys - ypred) .< params.eps
        # num_inliers = sum(inliers)
    

    # A = hcat(x_inliers, ones(length(x_inliers)))
    # inlier_slope, inlier_intercept = A \ y_inliers


    
    # Make a noisy proposal.
    slope     ~ normal(inlier_slope, 0.5)
    intercept ~ normal(inlier_intercept, 0.5)
    # We return values here for testing; normally, proposals don't have to return values.
    return inlier_slope, inlier_intercept
end;

function inlier_heuristic_update(tr)
    # Use inlier heuristics to (potentially) jump to a better line
    # from wherever we are.


    #HERE is initialization exercise. delete this line 
    (tr, _) = mh(tr, inlier_heuristic_proposal, (xs, ys))    
    #jumps a little crazier without this line 


    # Spend a while refining the parameters, using Gaussian drift
    # to tune the slope and intercept, and resimulation for the noise
    # and outliers.
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
end

#fake stuff is irrelevant and not used in the end 
fakeys = Float64[]

for ind in 1:20
    push!(fakeys, (rand(Float64)-0.5)*10)

end

#println(fakeys)

fakeobservations = make_constraints(fakeys); 

#NORMAL 
tr, = Gen.generate(regression_with_outliers, (xs,), observations)
#FAKE
faketr, = Gen.generate(regression_with_outliers, (xs,), fakeobservations)


#println(tr)
viz = @animate for i in 1:100
    global tr
    tr = inlier_heuristic_update(tr)
    visualize_trace(tr; title="Iteration $i")
end
gif(viz)

#println(observations)
#println(ys)

#initialization exercise above the 20 loop 


