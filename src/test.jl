
module Foo
using Gen

@gen function foo()
    x = {:x} ~ normal(30, 1)
    x
end

end # module Foo
