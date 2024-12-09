
using LinearAlgebra
using JuMP

abstract type FunnelConstraint end

struct StateConstraint <: FunnelConstraint
    a::Vector
    b::Float64
    function StateConstraint(a::Vector,b::Float64)
        new(a,b)
    end
end

struct InputConstraint <: FunnelConstraint
    a::Vector
    b::Float64
    function InputConstraint(a::Vector,b::Float64)
        new(a,b)
    end
end

function impose(constraint::StateConstraint,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector;idx::Int=-1)
    a = constraint.a
    b = constraint.b

    LMI = [(b-a'*xnom)*(b-a'*xnom) a'*Q;
        Q*a Q
    ]
    return LMI
end

function impose(constraint::InputConstraint,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector;idx::Int=-1)
    a = constraint.a
    b = constraint.b

    LMI = [(b-a'*unom)*(b-a'*unom) a'*Y;
        Y'*a Q
    ]
    return LMI
end

struct ObstacleAvoidance <: FunnelConstraint
    H::Matrix
    c::Vector
    function ObstacleAvoidance(H::Matrix,c::Vector)
        new(H,c)
    end
end

function impose(constraint::ObstacleAvoidance,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector;idx::Int=-1)
    H = constraint.H
    c = constraint.c
    M = [1 0 0;0 1 0]
    a = - M'*H'*H*(M*xnom-c) / norm(H*(M*xnom-c))
    s = 1 - norm(H*(M*xnom-c))
    b = -s + a'*xnom

    LMI = [(b-a'*xnom)*(b-a'*xnom) a'*Q;
        Q*a Q
    ]
    return LMI
end

# struct WayPoint <: FunnelConstraint
#     Qpos_max::Matrix{Float64}
#     function WayPoint(Qmax::Matrix)
#         new(Qmax)
#     end
# end

# function impose!(constraint::WayPoint,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector,idx::Int)
#     # hard coding
#     # 1,4,7,10,13,16
#     if (idx == 4) || (idx == 7) || (idx == 10) || (idx == 13)
#         @constraint(model, Q[1:3,1:3] <= constraint.Qpos_max, PSDCone())
#     end
# end