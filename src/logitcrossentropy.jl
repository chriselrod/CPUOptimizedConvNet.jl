
struct LogitCrossEntropy end

@generated function loss(::LogitCrossEntropy, x::AbstractStrideMatrix{N,M,T}, y) where {M,N,T}
    quote
        l = zero($T)
        @avx for n ∈ axes(x,1)
            xₙ_1 = x[n,1]
            Base.Cartesian.@nexprs $(M-1) m -> xₙ_{m+1} = max(x[n,m+1], xₙ_m)
            s_1 = exp(x[n,1] - $(Symbol(:xₙ_,M)))
            Base.Cartesian.@nexprs $(M-1) m -> s_{m+1} = s_m + exp(x[n,m+1] - $(Symbol(:xₙ_,M)))
            logspxmax = log($(Symbol(:s_,M))) + $(Symbol(:xₙ_,M))
            Base.Cartesian.@nexprs $M m -> l -= (y[n] == (m-1)) * (x[n,m] - logspxmax)
        end
        l
    end
end
function logit_cross_entropy_grad_expr(loss, M, ::Type{T}) where {T}
    loopbody = quote
        xₙ_1 = x[n,1]
        Base.Cartesian.@nexprs $(M-1) m -> xₙ_{m+1} = max(x[n,m+1], xₙ_m)
        s_1 = exp(x[n,1] - $(Symbol(:xₙ_,M)))
        Base.Cartesian.@nexprs $(M-1) m -> s_{m+1} = s_m + exp(x[n,m+1] - $(Symbol(:xₙ_,M)))
        logspxmax = log($(Symbol(:s_,M))) + $(Symbol(:xₙ_,M))
        Base.Cartesian.@nexprs $M m -> δx_m = (x[n,m] - logspxmax)
        Base.Cartesian.@nexprs $M m -> y_m = y[n] == (m-1)
    end
    loss && push!(loopbody.args, :(Base.Cartesian.@nexprs $M m -> l -= y_m ? δx_m : zero($T)))
    push!(loopbody.args, :(Base.Cartesian.@nexprs $M m -> ∂x[n,m] = exp(δx_m) - y_m))
    quote
        l = zero($T)
        @avx for n ∈ axes(∂x,1)
            $loopbody
        end
        l
    end    
end
@generated function gradient!(∂x, ::LogitCrossEntropy, x::AbstractStrideMatrix{N,M,T}, y) where {M,N,T}
    logit_cross_entropy_grad_expr(false, M, T)
end
@generated function loss_and_gradient!(∂x, ::LogitCrossEntropy, x::AbstractStrideMatrix{N,M,T}, y) where {M,N,T}
    logit_cross_entropy_grad_expr(true, M, T)
end


