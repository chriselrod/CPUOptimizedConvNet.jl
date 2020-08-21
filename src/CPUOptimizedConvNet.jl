module CPUOptimizedConvNet

using LoopVectorization, PaddedMatrices
using PaddedMatrices.VectorizationBase: maybestaticsize, maybestaticfirst, maybestaticlast, staticm1, staticp1

export clippedimageview, zero_edges!

include("utils.jl")
# include("optimize.jl")
include("convolution.jl")
include("maxpool.jl")
include("logitcrossentropy.jl")

end
