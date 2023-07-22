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


function copy_data()
    for game in filter(x -> occursin("-v5",x), readdir("out/gameplay"))
        mkpath("atari-benchmarks/variety/$game")
        for t in 200:220
            cp("out/gameplay/$game/$t.png", "atari-benchmarks/variety/$game/$t.png")
        end
    end
end

include("html.jl")

function make_quilt()
    all_frames = Array{RGB{Float64}}[]
    for gamepath in filter(x -> occursin("-v5",x), readdir("out/gameplay",join=true))
        println("processing $gamepath")
        files = filter(x -> endswith(x,".png"), readdir(gamepath))
        sort!(files, by = f -> parse(Int, split(f, ".")[1]))
        files = files[200:600-1]
        frames = stack([load(joinpath(gamepath,f)) for f in files], dims=3)
        if size(frames) != (210,160,400) println("skipping because size=$(size(frames))"); continue end
        push!(all_frames, frames)
    end

    WIDTH = 10

    println("adding padding")
    for _ in 1:(WIDTH - length(all_frames) % WIDTH)
        push!(all_frames, fill(RGB(0.,0.,0.),210,160,400))
    end

    # gif = zeros(210,160,1000)

    # for row in 1:div(length(all_frames), WIDTH)
    #     for col in 1:WIDTH
    #         i = (row-1)*WIDTH + col
    #         all_frames[i] = all_frames[i][:,:,1:100]
    #     end
    # end

    

    # @show size(all_frames)
    # @show typeof(all_frames)

    # @show size(reshape(all_frames, WIDTH, :))

    println("hvcat")
    res = hvcat(WIDTH, all_frames...);
    println("saving gif")
    fresh(); html_body(html_gif(res,width="1000px")); render()
end

