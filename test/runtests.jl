using CPUOptimizedConvNet
using Test
# Additional test deps
using ForwardDiff, PaddedMatrices, LinearAlgebra, Random

const H, W = 10, 10

@testset "CPUOptimizedConvNet.jl" begin
    N = 30;
    kern_unpermed = @FixedSize rand(Float32, 3, 3, 3, 16);
    img = zero_edges!(rand!(StrideArray{Float32}(undef, (Static(H+2), Static(W+2), Static(3), N))));
    # out1 = StrideArray{Float32}(undef, (Static(H),Static(W),Static(16),N));

    out = PermutedDimsArray(StrideArray{Float32}(undef, (Static(16),Static(H),Static(W),N)), (2,3,1,4));
    kern = PermutedDimsArray(FixedSizeArray{Tuple{16,3,3,3},Float32}(undef),(2,3,4,1)); kern .= kern_unpermed;

    @time CPUOptimizedConvNet.convlayeravx!(out, img, kern);
    
    imga = Array(img);
    kerna = Array(kern_unpermed);
    outa = Array(out);

    A = zero_edges!(PermutedDimsArray(rand!(StrideArray{Float32}(undef, (Static(16), Static(H+2), Static(W+2), N))), Static((2,3,1,4))));

    
    gimga, gkerna = let A = Array(clippedimageview(A, kern)), kerna = kerna, imga = imga
        f(img) = dot(A, CPUOptimizedConvNet.convlayeravx(img, kerna))
        gimg = @time ForwardDiff.gradient(f, imga, ForwardDiff.GradientConfig(f, imga, ForwardDiff.Chunk(16)))[begin+1:end-1,begin+1:end-1,:,:]
        g(kern) = dot(A, CPUOptimizedConvNet.convlayeravx(imga, kern))
        gkern = @time ForwardDiff.gradient(g, kerna, ForwardDiff.GradientConfig(g, kerna, ForwardDiff.Chunk(16)))
        gimg, gkern
    end;

    gimg = StrideArray{Float32}(undef, (Static(H), Static(W), Static(3), N));
    @time CPUOptimizedConvNet.convlayeradjimg!(gimg, kern, A);
    @test gimg ≈ gimga
    
    gkern = similar(kern);
    @time CPUOptimizedConvNet.convlayeradjkern!(gkern, img, clippedimageview(A, kern));
    @test gkern ≈ gkerna

    
    pooled = PermutedDimsArray(StrideArray{Float32}(undef, (Static(16),Static(H>>1),Static(W>>1),N)), Static((2,3,1,4)));
    B = PermutedDimsArray(rand!(StrideArray{Float32}(undef, (Static(16),Static(H>>1),Static(W>>1),N))), Static((2,3,1,4)));

    gpooled = PermutedDimsArray(StrideArray{Float32}(undef, (Static(16),Static(H),Static(W),N)), (2,3,1,4));
    CPUOptimizedConvNet.maxpool2x2relureversev2!(copyto!(gpooled, out), B)
    # CPUOptimizedConvNet.maxpool2x2relureversev2!(copyto!(gpooled, out), B)

    # Bp = PermutedDimsArray(rand!(StrideArray{Float32}(undef, (Static(16),Static(H>>1),Static(W>>1),N))), Static((2,3,1,4)));
    # CPUOptimizedConvNet.maxpool2x2relu!(Bp, out)
    # @benchmark CPUOptimizedConvNet.maxpool2x2relureverse!(copyto!($gpooled, $out), $B, $Bp)
    
    gmaxpool = let Ba = Array(B), outa = outa;
        f(x) = dot(Ba, CPUOptimizedConvNet.maxpool2x2relu!(similar(x, size(Ba)), x))
        @time ForwardDiff.gradient(f, outa, ForwardDiff.GradientConfig(f, outa, ForwardDiff.Chunk(16)))
    end;

    @test gmaxpool == gpooled

    y = rand(Int32(0):Int32(9), N);
    #    Bf = PaddedMatrices.flatten(B);
    C = randn!(StrideMatrix{Float32}(undef, (N, Static(10))));
    
    
    glossa, loss = let Ca = Array(C'), y = 0:9 .== y'
        # https://github.com/FluxML/NNlib.jl/blob/d524be7dc1a44c96793c0e9b420469204a85e9e7/src/softmax.jl#L85
        function logsoftmax(xs::AbstractArray; dims=1)
            max_ = maximum(xs, dims=dims)
            exp_ = exp.(xs .- max_)
            log_ = log.(sum(exp_, dims=dims))
            (xs .- max_) .- log_
        end
        # https://github.com/FluxML/Flux.jl/blob/2ea0d3581fa8f91de7ba05639734ad39870500d7/src/losses/functions.jl#L72-L84
        function logitcrossentropy(ŷ, y; dims=1, agg=sum)#mean)
            agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims))
        end
        lce(ŷ) = logitcrossentropy(ŷ, y)
        g = @time ForwardDiff.gradient(lce, Ca, ForwardDiff.GradientConfig(lce, Ca, ForwardDiff.Chunk(16)))
        l = lce(Ca)
        g, l
    end

    gloss1 = StrideMatrix{Float32}(undef, (N, Static(10)));
    @test loss ≈ CPUOptimizedConvNet.loss_and_gradient!(gloss1, CPUOptimizedConvNet.LogitCrossEntropy(), C, y)
    @test gloss1 ≈ glossa'
    gloss2 = StrideMatrix{Float32}(undef, (N, Static(10)));
    CPUOptimizedConvNet.gradient!(gloss2, CPUOptimizedConvNet.LogitCrossEntropy(), C, y)
    @test gloss2 ≈ glossa'
    @test loss ≈ CPUOptimizedConvNet.loss(CPUOptimizedConvNet.LogitCrossEntropy(), C, y)
end
