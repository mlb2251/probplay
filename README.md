# probabilistic-atari

## setup
Make sure you're using the julia environment associated with this project

## example of importing and calling things
```
using Atari
fresh(); particle_filter(5, crop(load_frames("atari-benchmarks/frostbite_1"), top=145, bottom=45, left=90, tskip=4)[:,:,:,1:4], 8); render();
```

## Run tests
```
]test
```

## Run examples
```
includet("examples/frostbite.jl")
fresh(); full1() render();
```
