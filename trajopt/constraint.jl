using LinearAlgebra
using JuMP

abstract type Constraint end

struct InputLinear <: Constraint
    A::Matrix
    b::Vector
    function InputLinear(A::Matrix,b::Vector)
        new(A,b)
    end
end

function impose!(constraint::InputLinear,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing])
    A = constraint.A
    b = constraint.b
    # return A * u == b
    @constraint(model, A*u .<= b) 
end

struct Obstacle <: Constraint
    H::Matrix
    c::Vector
    function Obstacle(H::Matrix,c::Vector)
        new(H,c)
    end
end

function impose!(constraint::Obstacle,model::Model,x::Vector,u::Vector,xbar::Vector,ubar::Vector=[nothing])
    # ||H(r-c)|| >= 1

    # obstacle parameters
    H = constraint.H
    c = constraint.c

    # position in state
    r = x[1:2]
    rbar = xbar[1:2]
    
    norm_H_rbar_c = norm(H*(rbar-c)) # ||H(rbar-c)||
    sbar = 1 - norm_H_rbar_c
    dsbar = - H'*H*(rbar-c) / norm_H_rbar_c

    A = dsbar'
    b = -sbar + dsbar' * rbar
    @constraint(model, A*r .<= b) 
end

struct PDG <: Constraint
    m_dry::Float64

    vmax::Float64
    wmax::Float64

    gamma_s::Float64
    theta_max::Float64

    Fmin::Float64
    Fmax::Float64
    tau_max::Float64
    delta_max::Float64
    function PDG()
        m_dry = 750

        vmax = 90
        wmax = deg2rad(5)
        glide_slope_max = deg2rad(20) 
        theta_max = deg2rad(90)

        Fmin = 600
        Fmax = 3000
        tau_max = 50
        delta_max = deg2rad(20)

        new(m_dry,vmax,wmax,glide_slope_max,theta_max,Fmin,Fmax,tau_max,delta_max)
    end
end

function impose!(pdg::PDG,model::Model,x::Vector,u::Vector,xbar::Vector=[nothing],ubar::Vector=[nothing])
    # m rx ry rz vx vy vz roll pitch yaw wx wy wz
    # 1  2  3  4  5  6  7 . 8 . 9    10  11 12 13
    # mass
    m = x[1]
    @constraint(model,pdg.m_dry <= m)

    # maximum velocity
    v = x[5:7]
    @constraint(model,[pdg.vmax;v] in SecondOrderCone())

    # maximum angular velocity
    w = x[11:13]
    @constraint(model,[pdg.wmax;w] in SecondOrderCone())

    # glide slope angle
    @constraint(model, [x[4]/tan(pdg.gamma_s); x[2:3]] in SecondOrderCone())

    # maximum tilt
    roll = x[8]
    pitch = x[9]
    @constraint(model, [roll,-roll] .<= [pdg.theta_max;pdg.theta_max])
    @constraint(model, [pitch,-pitch] .<= [pdg.theta_max;pdg.theta_max])

    # minimum thrust (non convex)
    F = u[1:3]
    Fbar = ubar[1:3]
    @constraint(model, pdg.Fmin - Fbar'*F / norm(Fbar,2) <= 0)

    # maximum thrust
    @constraint(model, [pdg.Fmax;F] in SecondOrderCone())

    # gimbal angle
    @constraint(model, [u[3]/cos(pdg.delta_max);F] in SecondOrderCone())

    # maximum torque
    T = u[4:6]
    # @constraint(model, [pdg.tau_max;T] in SecondOrderCone())
    @constraint(model, [pdg.tau_max; T] in MOI.NormInfinityCone(1 + length(T)))
end

function initial_condition!(dynamics::Dynamics,model::Model,x1::Vector,xi::Vector)
    @constraint(model,x1 == xi)
end
function final_condition!(dynamics::Unicycle,model::Model,xN::Vector,xf::Vector)
    @constraint(model,xN == xf)
end
function final_condition!(dynamics::Rocket,model::Model,xN::Vector,xf::Vector)
    @constraint(model,xN[2:dynamics.ix] == xf[2:dynamics.ix])
end