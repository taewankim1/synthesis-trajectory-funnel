
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

function impose!(constraint::StateConstraint,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector)
    a = constraint.a
    b = constraint.b

    LMI = [(b-a'*xnom)*(b-a'*xnom) a'*Q;
        Q*a Q
    ]
    @constraint(model, 0 <= LMI, PSDCone())
end

function impose!(constraint::InputConstraint,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector)
    a = constraint.a
    b = constraint.b

    LMI = [(b-a'*unom)*(b-a'*unom) a'*Y;
        Y'*a Q
    ]
    @constraint(model, 0 <= LMI, PSDCone())
end

struct ObstacleAvoidance <: FunnelConstraint
    H::Matrix
    c::Vector
    function ObstacleAvoidance(H::Matrix,c::Vector)
        new(H,c)
    end
end

function impose!(constraint::ObstacleAvoidance,model::Model,Q::Matrix,Y::Matrix,xnom::Vector,unom::Vector)
    H = constraint.H
    c = constraint.c
    M = [1 0 0;0 1 0]
    a = - M'*H'*H*(M*xnom-c) / norm(H*(M*xnom-c))
    s = 1 - norm(H*(M*xnom-c))
    b = -s + a'*xnom

    LMI = [(b-a'*xnom)*(b-a'*xnom) a'*Q;
        Q*a Q
    ]
    @constraint(model, 0 <= LMI, PSDCone())
end
