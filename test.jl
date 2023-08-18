using Gen 
import Distributions


mydist = Gen.normal
myargs = [0, 1]


@gen function tester(mydist::Distribution, myargs)
    bla ~ mydist(myargs...)
    return bla
end 

