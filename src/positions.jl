using Revise
using Gen
using GenParticleFilters
using GenSMCP3


#way to test with just positions
# (all 1 object)
#given list of positions, try sampling and scoring code prios to match 


#example positions to match 
TEST_POSITIONS = Vec[Vec(1.0, 1.0), 
Vec(2.0, 3.0), 
Vec(3.0, 5.0), 
Vec(4.0, 7.0),
Vec(5.0, 9.0),
]

#mini particle filter 
function position_particle_filter(num_particles::Int, observed_images::Array{Float64,4}, num_samples::Int; mh_steps_init=800, mh_steps=10, floodfill=true, perception_mh=false)
    C,H,W,T = size(observed_images)
    init_obs = if floodfill
        (cluster, objs, sprites) = process_first_frame(observed_images[:,:,:,1])
         build_init_obs(H,W, sprites, objs, observed_images[:,:,:,1])
    else
        choicemap((:init => :observed_image, observed_images[:,:,:,1]))
    end
    state = pf_initialize(model, (H,W,1), init_obs, num_particles)

    elapsed=@elapsed for t in 1:T-1
        @show t
        obs = choicemap((:steps => t => :observed_image, observed_images[:,:,:,t+1]))

        # take SMCP3 step
        @time pf_update!(state, (H,W,t+1), (NoChange(),NoChange(),UnknownChange()),
            obs, SMCP3Update(
                code_fwd_proposal,
                bwd_proposal_naive,
                (obs,),
                (obs,),
                false, # check are inverses
            ))        

        #do any html stuff here 

    
        (_, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
        weights = exp.(log_normalized_weights)
    end 
    
    return sample_unweighted_traces(state, num_samples)
end 



@kernel function code_fwd_proposal(prev_trace, obs)
    (H,W,prev_T) = get_args(prev_trace)
    t = prev_T # `t` will be our newly added timestep
    observed_image = obs[(:steps => t => :observed_image)]
    curr_env = deepcopy(env_of_trace(prev_trace))

    """
    t=0 is the first observation so t=1 is the second observation (after the first step)
    `T` is the total number of *observations* (`T-1` is the number of steps)
    """

    bwd_choices = choicemap()

    #current actual best positions
    positions = TEST_POSITIONS[1:t] #not sure abt this 

    #only on the 3rd object lol
    obj_id = 3 

    cfunc_options = []

    for j in 1:SAMPLES_PER_OBJ
        env = deepcopy(curr_env)
        cfunc = {:cfunc} ~ sample_cfunc(obj_id)
        cfunc = get_retval(cfunc.trace) #weird code feature to do this, cfunc not actually returned above 
        push!(cfunc_options, cfunc) #not sure bezt way to do this, maybe put j in the trace 
        step_of_obj = env.step_of_obj[obj_id]
        env.code_library[step_of_obj] = cfunc 
        
        #running the function once to see what positions it creates 
        poss_dyn_from_cfunc ~ obj_dynamics(obj_id, env, state_of_trace(prev_trace, 0), choicemap())
        #oh noooo the states arent stored anywhere where do i get all the positions

        #:step
        @show state.objs[obj_id].pos 

        #what about different times though 


        
        # env.state 
        # @show poss_dyn_from_cfunc
        #@show poss_dyn_from_cfunc





        #just one run of the code to get postiitons hmm 


    end 




    # sample different codes for an object 


    return(
        trace_updates, 
        bwd_choices
    )

end 