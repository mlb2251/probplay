import Random, Logging
using Gen, Plots
import StatsBase
include("regression_viz_import.jl")
include("gen_ii.jl")


#both (sometimes depending on initialization) explore the other slope, drift explores it more and at later iterations than map
#sometimes they never do becuase they randomly started on the more full slope 
#sometimes they land on the less common slope. Drift switches back and forth


(slope, intercept) = ransac(xs, ys_bimodal, RANSACParams(10, 3, 1.))
slope_intercept_init = choicemap()
slope_intercept_init[:slope] = slope
slope_intercept_init[:intercept] = intercept
(tr,) = generate(
    regression_with_outliers, (xs,), 
    merge(observations_bimodal, slope_intercept_init))

tr_drift = tr
tr_map   = tr

viz = Plots.@animate for i in 1:305
    global tr_map, tr_drift
    if i < 6
        tr_drift = ransac_update(tr)
        tr_map   = tr_drift
    else
        # Take a single gradient step on the line parameters.
        tr_map = map_optimize(tr_map, select(:slope, :intercept), max_step_size=1., min_step_size=1e-5)
        tr_map = map_optimize(tr_map, select(:noise), max_step_size=1e-2, min_step_size=1e-5)
        # Choose the most likely classification of outliers.
        tr_map = is_outlier_map_update(tr_map)
        # Update the prob outlier
        optimal_prob_outlier = mean([tr_map[:data => i => :is_outlier] for i in 1:length(xs)])
        optimal_prob_outlier = min(0.5, max(0.05, optimal_prob_outlier))
        tr_map, = update(tr_map, (xs,), (NoChange(),), choicemap(:prob_outlier => optimal_prob_outlier))
    
        # Gaussian drift update:
        tr_drift = gaussian_drift_update(tr_drift)
    end
    
    Plots.plot(visualize_trace(tr_drift; title="Drift (Iter $i)"), visualize_trace(tr_map; title="MAP (Iter $i)"))
end

drift_final_score = get_score(tr_drift)
map_final_score = get_score(tr_map)
println("i.   MH Gaussian drift score $(drift_final_score)")
println("ii.  MAP final score: $(final_score).")

gif(viz)








#counting how much pos/neg slope 
total_runs = 25;

for (index, value) in enumerate([(1, 0), (-1, 0), ransac(xs, ys_bimodal, RANSACParams(10, 3, 1.))])
    n_pos_drift = n_neg_drift = n_pos_map = n_neg_map = 0
    
    for i=1:total_runs
        pos_drift = neg_drift = pos_map = neg_map = false

        #### RANSAC for initializing
        (slope, intercept) = value # ransac(xs, ys_bimodal, RANSACParams(10, 3, 1.))
        slope_intercept_init = choicemap()
        slope_intercept_init[:slope] = slope
        slope_intercept_init[:intercept] = intercept
        (tr,) = generate(
            regression_with_outliers, (xs,), 
            merge(observations_bimodal, slope_intercept_init))
        for iter=1:5
            tr = ransac_update(tr)
        end
        ransac_score = get_score(tr)
        tr_drift = tr # version of the trace for the Gaussian drift algorithm
        tr_map = tr   # version of the trace for the MAP optimization

        #### Refine the parameters according to each of the algorithms
        for iter = 1:300
            # MAP optimiztion:
            # Take a single gradient step on the line parameters.
            tr_map = map_optimize(tr_map, select(:slope, :intercept), max_step_size=1., min_step_size=1e-5)
            tr_map = map_optimize(tr_map, select(:noise), max_step_size=1e-2, min_step_size=1e-5)
            # Choose the most likely classification of outliers.
            tr_map = is_outlier_map_update(tr_map)
            # Update the prob outlier
            optimal_prob_outlier = count(i -> tr_map[:data => i => :is_outlier], 1:length(xs)) / length(xs)
            optimal_prob_outlier = min(0.5, max(0.05, optimal_prob_outlier))
            (tr_map, _) = update(tr_map, (xs,), (NoChange(),), choicemap(:prob_outlier => optimal_prob_outlier))

            # Gaussian drift update:
            tr_drift = gaussian_drift_update(tr_drift)

            if tr_drift[:slope] > 0
                pos_drift = true
            elseif tr_drift[:slope] < 0
                neg_drift = true
            end
            if tr_map[:slope] > 0
                pos_map = true  
            elseif tr_map[:slope] < 0
                neg_map = true
            end

        end

        if pos_drift
            n_pos_drift += 1
        end
        if neg_drift
            n_neg_drift += 1
        end
        if pos_map
            n_pos_map += 1
        end
        if neg_map
            n_neg_map += 1
        end
    end
    (slope, intercept) = value
    println("\n\nWITH INITIAL SLOPE $(slope) AND INTERCEPT $(intercept)")
    println("TOTAL RUNS EACH: $(total_runs)")
    println("\n       times neg. slope    times pos. slope")
    println("\ndrift: $(n_neg_drift)                  $(n_pos_drift)")
    println("\nMAP:   $(n_neg_map)                    $(n_pos_map)")
end