
struct DenseT{M,K,T,X1,X2,L}
    A::FixedSizeMatrix{M,K,T,X1,X2,L}
end

function forward!(C, d::DenseT, B)
    @unpack A = d
    Koff = maybestaticsize(A, Val{2}())
    @avx for n ∈ axes(C,2), m ∈ axes(C,1)
        Cₘₙ = zero(eltype(C))
        for k ∈ axes(B,1)
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = Cₘₙ + A[m, Koff]
    end
end
function reverse_grad!(Ā, C̄, B)
    @avx for k ∈ axes(B,1), m ∈ axes(Ā,1)
        Āₘₖ = zero(eltype(Ā))
        for n ∈ axes(C̄,2)
            Āₘₖ += C̄[m,n] * B[k,n]
        end
        Ā[m,k] = Āₘₖ# + A[m, Koff]
    end
    Koff = maybestaticsize(Ā, Val{2}())
    @avx for m ∈ axes(Ā,1), n ∈ axes(C̄,2)
        A[m,Koff] += C̄[m,n]
    end
end
function reverse_chain!(B̄, A, C̄)
    @avx for k ∈ axes(B̄,1), n ∈ axes(B̄2)
        B̄ₖₙ = zero(eltype(B̄))
        for m ∈ axes(A,1)
            B̄ₖₙ += A[m,k] * C̄[m,n]
        end
        B̄[k,n] = B̄ₖₙ
    end
end

