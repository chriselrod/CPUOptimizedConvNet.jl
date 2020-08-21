function maxpool2x2relu!(B, A)
    @avx for i₁ ∈ axes(B,1), i₂ ∈ axes(B,2), i₃ ∈ axes(B,3), i₄ ∈ axes(B,4)
        A₁ = A[2i₁-1,2i₂-1,i₃,i₄]
        A₂ = A[2i₁-1,2i₂  ,i₃,i₄]
        A₃ = A[2i₁  ,2i₂-1,i₃,i₄]
        A₄ = A[2i₁  ,2i₂  ,i₃,i₄]
        B[i₁,i₂,i₃,i₄] = max(max(max(A₁, A₂), max(A₃, A₄)), zero(eltype(A)))
    end
    B
end
function maxpool2x2relureverse!(A, B̄, B)
    @avx unroll=(1,1) for i₁ ∈ axes(B,1), i₂ ∈ axes(B,2), i₃ ∈ axes(B,3), i₄ ∈ axes(B,4)
        Bₘ = B[i₁,i₂,i₃,i₄]
        B̄ᵢ = B̄[i₁,i₂,i₃,i₄]
        A[2i₁-1,2i₂-1,i₃,i₄] = (A[2i₁-1,2i₂-1,i₃,i₄] == Bₘ) * B̄ᵢ
        A[2i₁-1,2i₂  ,i₃,i₄] = (A[2i₁-1,2i₂  ,i₃,i₄] == Bₘ) * B̄ᵢ
        A[2i₁  ,2i₂-1,i₃,i₄] = (A[2i₁  ,2i₂-1,i₃,i₄] == Bₘ) * B̄ᵢ
        A[2i₁  ,2i₂  ,i₃,i₄] = (A[2i₁  ,2i₂  ,i₃,i₄] == Bₘ) * B̄ᵢ
    end
end
function maxpool2x2relureversev2!(A, B̄)
    @avx unroll=(1,1) for i₁ ∈ axes(B̄,1), i₂ ∈ axes(B̄,2), i₃ ∈ axes(B̄,3), i₄ ∈ axes(B̄,4)
        A₁ = A[2i₁-1,2i₂-1,i₃,i₄]
        A₂ = A[2i₁-1,2i₂  ,i₃,i₄]
        A₃ = A[2i₁  ,2i₂-1,i₃,i₄]
        A₄ = A[2i₁  ,2i₂  ,i₃,i₄]
        Bₘ = max(max(max(A₁, A₂), max(A₃, A₄)), zero(eltype(A)))
        B̄ᵢ = B̄[i₁,i₂,i₃,i₄]
        A[2i₁-1,2i₂-1,i₃,i₄] = (A₁ == Bₘ) * B̄ᵢ
        A[2i₁-1,2i₂  ,i₃,i₄] = (A₂ == Bₘ) * B̄ᵢ
        A[2i₁  ,2i₂-1,i₃,i₄] = (A₃ == Bₘ) * B̄ᵢ
        A[2i₁  ,2i₂  ,i₃,i₄] = (A₄ == Bₘ) * B̄ᵢ
    end
end

@generated function halve12(x::Tuple{Vararg{<:Any,N}}) where {N}
    out = Expr(:tuple)
    for n in 1:N
        r = Expr(:ref, :x, n)
        if n ≤ 2
            r = Expr(:call, :>>>, r, Expr(:call, Expr(:curly, :Static, 1)))
        end
        push!(out.args, r)
    end
    out
end
function default_alloc_maxpool(mp::MaxPool2x2Layer, img::AbstractArray{T}) where {T}
    s = halve12(maybestaticsize(img))
    _, p = PaddedMatrices.size_permute_tuples(img)
    B = allocarray(T, s)
    PermutedDimsArray(B, p)
end
function (sp::StatckPointer)(::typeof(default_alloc_maxpool), mp::MaxPool2x2Layer, img::AbstractArray{T}) where {T}
    s = halve12(maybestaticsize(img))
    _, p = PaddedMatrices.size_permute_tuples(img)
    sp, B = PtrArray{T}(sp, s)
    sp, PermutedDimsArray(B, p)
end
function alloc_maxpool_pad12(mp::MaxPool2x2Layer, img::AbstractArray{T,4}) where {T}
    strunc = halve12(maybestaticsize(img))
    s = (vadd(strunc[1], Static{2}()), vadd(strunc[2], Static{2}()), strunc[3], strunc[4])
    _, p = PaddedMatrices.size_permute_tuples(img)
    B = allocarray(T, s)
    PermutedDimsArray(B, p)
end
function (sp::StatckPointer)(::typeof(default_alloc_maxpool), mp::MaxPool2x2Layer, img::AbstractArray{T,4}) where {T}
    s = halve12(maybestaticsize(img))
    _, p = PaddedMatrices.size_permute_tuples(img)
    sp, B = PtrArray{T}(sp, s)
    sp, PermutedDimsArray(B, p)
end

struct MaxPool2x2Layer{F} <: AbstractLayer
    f::F
end
MaxPool2x2Layer() = MaxPool2x2Layer(default_alloc_maxpool)

parameters(::MaxPool2x2Layer) = nothing
grad(::MaxPool2x2Layer) = nothing
returns(mp::MaxPool2x2Layer) = mp.o



function forward(sp::StackPointer, mp::MaxPool2x2Layer, img)

    sp, out = stack_pointer_call(mp.f, sp, img)

    maxpool2x2relu!(out, img)
    out, pb, out
end
forward!(mp::MaxPool2x2Layer, img) = (maxpool2x2relu!(returns(mp), img), img)
reverse_grad!(::MaxPool2x2Layer) = nothing
function reverse_chain!(mp::MaxPool2x2Layer, img, ōūt̄)
    maxpool2x2relureversev2!(img, ōūt̄)
end

# function maxpool2x2relureversev3!(A, B̄)
#     for i₁ ∈ axes(B̄,1), i₂ ∈ axes(B̄,2), i₄ ∈ axes(B̄,4)
#         @inbounds @simd ivdep for i₃ ∈ axes(B̄,3)
#             A₁ = A[2i₁-1,2i₂-1,i₃,i₄]
#             A₂ = A[2i₁-1,2i₂  ,i₃,i₄]
#             A₃ = A[2i₁  ,2i₂-1,i₃,i₄]
#             A₄ = A[2i₁  ,2i₂  ,i₃,i₄]
#             Bₘ = max(max(max(A₁, A₂), max(A₃, A₄)), zero(eltype(A)))
#             B̄ᵢ = B̄[i₁,i₂,i₃,i₄]
#             A[2i₁-1,2i₂-1,i₃,i₄] = (A₁ == Bₘ) * B̄ᵢ
#             A[2i₁-1,2i₂  ,i₃,i₄] = (A₂ == Bₘ) * B̄ᵢ
#             A[2i₁  ,2i₂-1,i₃,i₄] = (A₃ == Bₘ) * B̄ᵢ
#             A[2i₁  ,2i₂  ,i₃,i₄] = (A₄ == Bₘ) * B̄ᵢ
#         end
#     end
# end


