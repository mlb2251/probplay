using Gen
using Distributions



struct Uniform_not_x <: Gen.Distribution{Int} end

function Gen.random(::Uniform_not_x, x, min, max)
    #get random int between min and max not xmax
    y = rand(min:max-1)
    if y >= x
        y += 1
    end
    return y
end

function Gen.logpdf(::Uniform_not_x, y, x, min, max)
    if y == x
        return -Inf
    end 
    if y >= min && y <= max
        return -log(max-min) #log of (1/(max-min))
    else 
        return -Inf
    end
end

const uniform_not_x = Uniform_not_x()
(::Uniform_not_x)(x, min, max) = random(Uniform_not_x(), x, min, max)





#make more extreme maybe??
function get_alpha(peak) 
    if peak <= 0
        return 0.00001
    elseif peak >= 1
        return 0.99999
    else
        return peak / (1 - peak)
    end
end

@dist beta_with_peak(peak) = beta(get_alpha(peak), 1)

