using Gen 
import Distributions


mydist = Gen.normal
myargs = [0, 1]


@gen function tester(mydist::Distribution, myargs)
    bla ~ mydist(myargs...)
    return bla
end 




test = [3, 4, 5]
testy = [test, test]

function hi()
    testy[begin:end,1]
end 

hi()