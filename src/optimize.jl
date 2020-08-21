
struct ADAM
    η::Float64
    β::Tuple{Float64,Float64}
end
struct ADAMState{A}
    mv::A
    vt::A
    βp::Base.RefValue{Tuple{Float64,Float64}}
end
ADAMState(

function update!(θ::AbstractArray{T}, s::ADAMState, Δ::AbstractArray{T}, a::ADAM) where {T}
    @unpack η, (β₁, β₂) = a
    @unpack mv, vt, βp = s
    βp₁, βp₂ = βp
    omβ₁ = T(1 - β₁); βp₁⁻¹ = T(inv(1 - βp₁))
    omβ₂ = T(1 - β₂); βp₂⁻ʰ = T(inv(sqrt(1 - βp₂)))
    @avx for i ∈ eachindex(Δ)
        Δₒ = Δ[i]
        mtₒ = mt[i]
        mtₙ = β₁ * mtₒ + omβ₁ * Δₒ
        vtₒ = vt[i]
        vtₙ = β₂ * vtₒ + omβ₂ * Δₒ * Δₒ
        Δₙ = mtₙ * βp₁⁻¹ / (sqrt(vtₙ)*βp₂⁻ʰ + ϵ)
        mt[i] = mtₙ
        vt[i] = vtₙ
        θ[i] -= Δₙ * η
    end
    s.βp[] = (βp₁*β₁, βp₂*β₂); nothing
end
update!(::Nothing, ::Nothing, ::Nothing, ::ADAM) = nothing

