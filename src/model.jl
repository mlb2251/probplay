using Images
using Gen
using Plots


struct Sprite
    mask::Matrix{Bool}
    color::RGB{Float64}
end

struct Ty
    sprites::Vector{Sprite}
end

struct Object
    type::Ty
    x::Int
    y::Int
    sprite_idx::Int
end

function canvas(height=210, width=160, background=RGB(0,0,0))
    fill(background, height, width)
end

function draw!(canvas, obj::Object)
    sprite = obj.type.sprites[obj.sprite_idx];
    for I in CartesianIndices(sprite.mask)
        i, j = Tuple(I)
        if sprite.mask[i,j]
            offy = obj.y+i-1
            offx = obj.x+j-1
            if offy > 0 && offy <= size(canvas,1) && offx > 0 && offx <= size(canvas,2)
                canvas[offy,offx] = sprite.color
            end
        end
    end
end

function draw!(canvas, objs::Vector{Object})
    for obj in objs
        draw!(canvas, obj)
    end
end


function sample_sprite()
    w = rand(1:100)
    h = rand(1:100)
    mask = rand(Bool, w, h)
    color = RGB(rand(), rand(), rand())
    Sprite(mask, color)
end

struct SpritePrior <: Distribution{Sprite} end

function random(::SpritePrior)
    w = rand(1:100)
    h = rand(1:100)
    mask = rand(Bool, w, h)
    color = RGB(rand(), rand(), rand())
    Sprite(mask, color)
end

function logpdf(::SpritePrior, sprite::Sprite)
    (1/100) * (1/100) * (1/2)^(size(sprite.mask,1) * size(sprite.mask,2))
end

const sprite_prior = SpritePrior()


@gen function model()
    sprite ~ sprite_prior()
    sprite
end





function sample_type()
    num_sprites = rand(1:3)
    sprites = [sample_sprite() for _ in 1:num_sprites]
    Ty(sprites)
end

function sample_obj(types, canvas)
    ty = rand(types)
    x = rand(1:size(canvas,2))
    y = rand(1:size(canvas,1))
    sprite_idx = rand(1:length(ty.sprites))
    Object(ty, x, y, sprite_idx)
end

function move_obj(obj, canvas)
    if rand() < .5
        return obj
    end
    x = clamp(obj.x + rand(-10:10), 1, size(canvas,2))
    y = clamp(obj.y + rand(-10:10), 1, size(canvas,1))
    Object(obj.type, x, y, obj.sprite_idx)
end

function change_sprite(obj)
    if rand() < .9
        return obj
    end
    sprite_idx = rand(1:length(obj.type.sprites))
    Object(obj.type, obj.x, obj.y, sprite_idx)
end


function test()
    c = canvas()

    num_types = rand(1:8)
    types = [sample_type() for _ in 1:num_types]

    num_objs = rand(1:16)
    objs = [sample_obj(types, c) for _ in 1:num_objs]

    # draw!(canvas(), objs)

    @gif for i in 1:100
        objs = [move_obj(obj, c) for obj in objs]
        objs = [change_sprite(obj) for obj in objs]
        c = canvas()
        draw!(c, objs)
        plot(c, xlims=(0,160), ylims=(0,210), ticks=true)
        annotate!(-60, 20, "Step: $i")
    end
end

function preview_games()
    frames = Matrix{RGB{N0f8}}[]
    for (dir,folders,files) in walkdir("out/gameplay")
        for file in files
            if "100.png" == file
                img = load("$dir/$file")
                if size(img) == (210,160)
                    push!(frames, img)
                end
            end
        end
    end
    mosaicview(frames, nrow=8)
end



# 210 x 160

function loop()
    for (dir,folders,files) in walkdir("out/gameplay")
        for file in files
            if "1.png" == file
                img = load("$dir/$file")
                println("$dir/$file ", size(img))
                
                # img = imresize(img, (84,84))
                # save("out/gameplay/$file", img)
            end
        end
    end
end