module Atari

using Gen

@gen function foo(input_arguments)
    x = {:initial_x} ~ normal(0, 1)
end

end # module Atari
