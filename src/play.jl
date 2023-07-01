# using ReinforcementLearning
# using ArcadeLearningEnvironment
using PyCall
using Images
# import gymnasium as gym
# env = gym.make('CartPole-v1')

function to_img(obs)
    Array(colorview(RGB,PermutedDimsArray(obs / 255, (3,1,2))))
end



function gen_gameplay(;game=nothing, human=false)
    gym = pyimport("gymnasium")
    play = pyimport("gymnasium.utils.play")
    #game = "ALE/Frostbite-v5"
    # game = "ALE/Boxing-v5"
    is_rand = isnothing(game)
    games = [x for x in keys(gym.envs.registry) if occursin("-v5",x) && occursin("ALE/",x) && !occursin("-ram-",x)]

    if human
        while true
            if is_rand
                game = rand(games)
            end
            println("Playing: $game")
            env = gym.make(game, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
            play.play(env, zoom=4, fps=30)
            sleep(3)
        end
        return
    end


    for game in games
        #game = "ALE/Frostbite-v5"
        # game = "ALE/Boxing-v5"

        println("Playing: $game")
        env = gym.make(game, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)

        imgs = Matrix{RGB{Float64}}[]

        observation, info = env.reset(seed=42)

        for i in 1:1000
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)

            obs = to_img(observation)
            push!(imgs, obs)

            if done
                observation, info = env.reset(seed=42)
            end
        end

        env.close()

        println("Saving images...")

        #game_dir = "out/gameplay/" * replace(game, "/" => "-", " " => "-")
        game_dir = "atari-benchmarks" * replace(game, "/" => "-", " " => "-")

        mkpath(game_dir)

        for (i,img) in enumerate(imgs)
            save("$game_dir/$i.png", img)
        end

    end
    println("Done")
end