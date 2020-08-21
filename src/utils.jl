
function zero_edges!(A::AbstractArray{T,4}) where {T}
    L₁ = last(axes(A,1)); L₂ = last(axes(A,2));
    if L₁ == L₂ && first(axes(A,1)) == first(axes(A,2))
        @avx for i₁ ∈ axes(A,2), i₃ ∈ axes(A,3), i₄ ∈ axes(A,4)
            A[i₁,1, i₃,i₄] = zero(T)
            A[i₁,L₁,i₃,i₄] = zero(T)
            A[1, i₁,i₃,i₄] = zero(T)
            A[L₁,i₁,i₃,i₄] = zero(T)
        end
    else
        for i₃ ∈ axes(A,3), i₄ ∈ axes(A,4)
            @avx for i₁ ∈ axes(A,1)
                A[i₁,1, i₃,i₄] = zero(T)
                A[i₁,L₂,i₃,i₄] = zero(T)
            end
            @avx for i₁ ∈ axes(A,2)
                A[1, i₁,i₃,i₄] = zero(T)
                A[L₁,i₁,i₃,i₄] = zero(T)
            end
        end
    end
    A
end

clippedrange(x) = staticp1(maybestaticfirst(x)):staticm1(maybestaticlast(x))
clippedrange(x, c) = maybestaticfirst(x)+c:maybestaticlast(x)-c

"""
For padded images, take a view to drop the padding. Used to pass image to maxpool2x2.
"""
@generated function clippedimageview(A::AbstractArray{<:Any,N}, ::AbstractArray) where {N}
    v = Expr(:call, :view, :A)
    for n in 1:N-2
        push!(v.args, :(clippedrange(axes(A,$n), maybestaticsize(A, Val{$n}()) >>> Static{1}())))
    end
    push!(v.args, :(:)); push!(v.args, :(:))
    Expr(:block, Expr(:meta,:inline), v)
end

unclippedrange(x) = staticm1(maybestaticfirst(x)):staticp1(maybestaticlast(x))
unclippedrange(x, c) = maybestaticfirst(x) - c:maybestaticlast(x) + c

@generated function unclip(A::AbstractArray{<:Any,N}, ::AbstractArray) where {N}
    v = Expr(:call, :view, :A)
    for n in 1:N
        if n < 3
            push!(v.args, :(unclippedrange(axes(A,$n), maybestaticsize(A, Val{$n}()) >>> Static{1}())))
        else
            push!(v.args, :(:))
        end
    end
    Expr(:block, Expr(:meta,:inline), v)
end

function alloc_maxpool2x2_output(img::AbstractArray{T}, ::Val{pad} = Val{true}()) where {T,pad}
    if pad
        H = vadd(PaddedMatrices.VectorizationBase.maybestaticsize(img, Val(1)) >> Static(1), Static(2))
        W = vadd(PaddedMatrices.VectorizationBase.maybestaticsize(img, Val(1)) >> Static(1), Static(2))
    else
        H = PaddedMatrices.VectorizationBase.maybestaticsize(img, Val(1)) >> Static(1)
        W = PaddedMatrices.VectorizationBase.maybestaticsize(img, Val(1)) >> Static(1)
    end
    O = PaddedMatrices.VectorizationBase.maybestaticsize(img, Val(3))
    N = PaddedMatrices.VectorizationBase.maybestaticsize(img, Val(4))
    A = allocarray(T, (O, H, W, N))
    P = PermutedDimsArray(A, Static((2,3,1,4)))
    padd ? zero_edges!(P) : P
end

