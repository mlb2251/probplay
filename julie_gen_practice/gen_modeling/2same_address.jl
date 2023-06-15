using Gen

@gen function trying_same_address() 
    mynorm = {:normal_address} ~ normal(100, 5)
    mynorm2 = {:normal_address} ~ normal(100, 5)
    print("same address:", mynorm-mynorm2)
    print("same var same address:", mynorm-mynorm)
end; 


trying_same_address()

#result, using the same address twice doesn't result in the same random variable
#unless it is named as a variable