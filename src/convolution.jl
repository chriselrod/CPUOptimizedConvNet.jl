
function convlayeravx!(out::AbstractArray{<:Any,4}, img, kern)
    @avx for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), d ∈ axes(out,4), o ∈ axes(kern,4)
        s = zero(eltype(out))
        for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), i ∈ axes(kern,3)
            s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[k₁, k₂, i, o]
        end
        out[j₁, j₂, o, d] = s
    end
    out
end

function convlayeradjkern!(kernadj::AbstractArray{<:Any,4}, img, outadj)
    @avx for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), i ∈ axes(kernadj,3), o ∈ axes(kernadj,4)
        s = zero(eltype(kernadj))
        for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), d ∈ axes(outadj,4)
            s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * outadj[j₁, j₂, o, d]
        end
        kernadj[k₁, k₂, i, o] = s
    end
    kernadj
end
@generated function convlayeradjimg!(imgadj, kern::AbstractStrideArray{Tuple{K₁,K₂,I,O},T,4}, outadj) where {K₁,K₂,I,O,T}
    quote
        for j₁ ∈ axes(imgadj,1), j₂ ∈ axes(imgadj,2), i ∈ axes(kern,3), d ∈ axes(outadj,4)
            s = zero($T)
            for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), o ∈ axes(kern,4)
                s += kern[k₁, k₂, i, o] * outadj[j₁ - k₁ + $K₁, j₂ - k₂ + $K₂, o, d]
            end
            imgadj[j₁, j₂, i, d] = s
        end
        imgadj
    end
end

function conv_output_size(img, kern)
    H = maybestaticsize(img, Val{1}())
    W = maybestaticsize(img, Val{2}())
    O = maybestaticsize(kern,Val{4}())
    D = maybestaticsize(img, Val{4}())
    (H, W, O, D)
end
function convlayeravx(img::AbstractArray{T}, kern) where {T}
    s = conv_output_size(img, kern)
    _, p = PaddedMatrices.size_permute_tuples(img)
    out = PermutedDimsArray(allocarray(T, s), p)
    convlayeravx!(out, img, kern)
end
function convlayeravx(sp::StackPointer, img::AbstractArray{T}, kern) where {T}
    s = conv_output_size(img, kern)
    _, p = PaddedMatrices.size_permute_tuples(img)
    sp, B = PtrArray{T}(sp, s)
    out = PermutedDimsArray(B, p)
    sp, convlayeravx!(out, img, kern)
end

struct ConvolutionLayer{Κ,Κ̄,O} <: AbstractLayer
    κ::Κ
    # κ̄::Κ̄
    # o::O
end
paramters(cl::ConvolutionLayer) = cl.κ
grad(cl::ConvolutionLayer) = cl.κ̄
returns(cl::ConvolutionLayer) = cl.o

# returns out, pullback, and preallocated ∂out
function forward(cl::ConvolutionLayer, img)
    out = convlayeravx(sp, img, cl.κ)
    out, img, out
end
function forward(sp::StackPointer, cl::ConvolutionLayer, img)
    sp, out = convlayeravx(sp, img, cl.κ)
    sp, (out, img, out)
end

# function forward!(cl::ConvolutionLayer, img)
#     out = returns(cl)
#     convlayeravx!(clippedimageview(returns(cl)), img, parameters(cl))
#     img, out, out
# end
function reverse_grad!(cl::ConvolutionLayer, img, ōūt̄)
    convlayeradjkern!(grad(cl), img, clippedimageview(ōūt̄))
end
function reverse_chain!(cl::ConvolutionLayer, img, ōūt̄)
    convlayeradjimg!(clippedimageview(img), parameters(cl), ōūt̄)
end

