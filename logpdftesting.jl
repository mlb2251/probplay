using Gen 
@gen function mattfunc()
    if({:x} ~ bernoulli(0.7))
        y ~ uniform(0, 0.6)
    else
        z ~ uniform(0, 0.1) # e^2.3 = 10 oml 
    end
end

tr, weight= generate(mattfunc, (), choicemap((:x, false))) # 
tr2, weight2 = generate(mattfunc, (), choicemap((:x, true)))
tr3, weight3 = generate(mattfunc, (), choicemap())
@show tr, tr2, tr3
@show weight, weight2, weight3
#@show mattfunc()
choices = get_choices(tr)
choices2 = get_choices(tr2)
choices3 = get_choices(tr3)
@show assess(mattfunc, (), choices)

@show assess(mattfunc, (), choices2)
@show assess(mattfunc, (), choices3)

@show update(tr, get_args(tr), (NoChange(),), choicemap((:x, false)))
tr2_updated, wt_update, retdiffs, discarded_things = update(tr2, get_args(tr), (NoChange(),), choicemap((:x, false)));
trz_updated, wtz, _, _ = update(tr2_updated, get_args(tr), (NoChange(),), choicemap((:z, 0.05)))#old arbitrary z choice and new have same weight 


trace_prob_diff =  assess(mattfunc, (), get_choices(tr2_updated))[1] - assess(mattfunc, (), get_choices(tr2))[1]

@show trace_prob_diff
@show wt_update #trace_prob_diff, (diff in log, divide in probs)

@show trace_prob_diff - wt_update

@show wtz



;
