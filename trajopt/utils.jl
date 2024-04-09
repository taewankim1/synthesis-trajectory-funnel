using Interpolations
using LinearAlgebra
# using Plots
using PyPlot
using DifferentialEquations
function print_jl(x,flag_val = false)
    println("Type is $(typeof(x))")
    println("Shape is $(size(x))")
    if flag_val == true
        println("Value is $(x)")
    end
end

include("dynamics.jl")
function matrix_to_vector(matrix::Array)
    return [vec(col) for col in eachcol(matrix)]
end

function propagate_multiple_FOH(model::Dynamics,x::Matrix,u::Matrix,T::Vector)
    N = size(x,2) - 1
    ix = size(x,1)
    iu = size(u,1)

    function model_wrapper!(f,x,p,t)
        um = p[1]
        up = p[2]
        dt = p[3]
        alpha = 1 - t
        beta = t
        u1 = alpha*um + beta*up
        f .= dt*forward(model,x,u1)
    end

    tspan = (0,1)
    tprop = []
    xprop = []
    xnew = zeros(size(x))
    xnew[:,1] .= x[:,1]
    for i in 1:N
        prob = ODEProblem(model_wrapper!,x[:,i],tspan,(u[:,i],u[:,i+1],T[i]))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);
        tode = sol.t
        xode = stack(sol.u)
        if i == 1
            tprop = T[i]*tode
            xprop = xode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+T[i]*tode)
            xprop = hcat(xprop,xode)
        end
        xnew[:,i+1] .= xode[:,end]
    end
    return xnew,tprop,xprop
end