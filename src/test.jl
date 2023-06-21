
module Foo
using Gen

@gen function foo()
    x = {:x} ~ normal(0, 1)
    x
end



end # module Foo
