"""
The `out` that a layer returns in `forward!` is allowed to be mutated, and thus can't be relied upon.
However, this does allow it to mutate the inputs.

`reverse_grad!` is called before `reverse_chain!`
"""
abstract type AbstractLayer end

struct Chain{D,C<:Tuple{Vararg{<:AbstractLayer,D}},O,F}
    chain::C
    optstates::NTuple{D,O}
    loss::F
end

@generated function (c::Chain{D})(x_0) where {D}
    quote
        Base.Cartesian.@nexprs $D d -> c_d = c.chain[d]
        Base.Cartesian.@nexprs $D d -> (x_d, xpb_d) = forward!(c_d, x_{d-1})
        loss(c.loss, $(Symbol(:x_, D)))
    end
end
@generated function gradient!(c::Chain{D}, x_0) where {D}
    x_D = Symbol(:x_,D);
    quote
        Base.Cartesian.@nexprs $D d -> c_d = c.chain[d]
        Base.Cartesian.@nexprs $D d -> (x_d, xpb_d) = forward!(c_d, x_{d-1})
        gradient!($x_D, c.loss, $x_D)
        Base.Cartesian.@nexprs $(D-1) d -> begin
            reverse_grad!(c_{$(D+1}-d}, xpb_{$(D+1}-d}, x_{$(D+1}-d})
            reverse_chain!(c_{$(D+1}-d}, xpb_{$(D+1}-d}, x_{$(D+1}-d})
        end
        reverse_grad!(c_1, xpb_1, x_1)
        (Base.Cartesian.@ntuple $D d -> grad(c_d))
    end
end

