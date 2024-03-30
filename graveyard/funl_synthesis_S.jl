using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel

include("funl_dynamics.jl")
include("funl_constraint.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/scaling.jl")

mutable struct FunnelSolution
    Q::Array{Float64,3}
    S::Array{Float64,3}
    Z::Array{Float64,3}

    Aq::Array{Float64,3}
    Bm::Array{Float64,3}
    Bp::Array{Float64,3}
    Sm::Array{Float64,3}
    Sp::Array{Float64,3}

    Qi::Matrix{Float64}
    Qf::Matrix{Float64}

    t::Vector{Float64}
    tprop::Any
    xprop::Any
    uprop::Any
    Qprop::Any
    Sprop::Any

    function FunnelSolution(N::Int64,ix::Int64,iu::Int64,iq::Int64,is::Int64)
        Q = zeros(ix,ix,N+1)
        S = zeros(iu,iu,N+1)
        Z = zeros(ix,ix,N+1)

        Aq = zeros(iq,iq,N)
        Bm = zeros(iq,is,N)
        Bp = zeros(iq,is,N)
        Sm = zeros(iq,iq,N)
        Sp = zeros(iq,iq,N)

        Qi = zeros(ix,ix)
        Qf = zeros(ix,ix)
        
        t = zeros(N+1)
        new(Q,S,Z,Aq,Bm,Bp,Sm,Sp,Qi,Qf,t)
    end
end

struct FunnelSynthesis
    dynamics::Dynamics
    funl_dynamics::FunnelDynamics
    funl_constraint::Vector{FunnelConstraint}
    scaling::Any
    solution::FunnelSolution

    N::Int64  # number of subintervals (number of node - 1)
    w_funl::Float64  # weight for funnel cost
    w_vc::Float64  # weight for virtual control
    w_tr::Float64  # weight for trust-region
    tol_tr::Float64  # tolerance for trust-region
    tol_vc::Float64  # tolerance for virtual control
    tol_dyn::Float64  # tolerance for dynamics error
    max_iter::Int64  # maximum iteration
    verbosity::Bool
    function FunnelSynthesis(N::Int,max_iter::Int,
        dynamics::Dynamics,funl_dynamics::FunnelDynamics,funl_constraint::Vector{T},scaling::Scaling,
        w_funl::Float64,w_vc::Float64,w_tr::Float64,tol_tr::Float64,tol_vc::Float64,tol_dyn::Float64,
        verbosity::Bool) where T <: FunnelConstraint
        ix = dynamics.ix
        iu = dynamics.iu
        iq = funl_dynamics.iq
        is = funl_dynamics.is
        solution = FunnelSolution(N,ix,iu,iq,is)
        new(dynamics,funl_dynamics,funl_constraint,scaling,solution,
            N,w_funl,w_vc,w_tr,tol_tr,tol_vc,tol_dyn,max_iter,verbosity)
    end

end

function sdpopt!(fs::FunnelSynthesis,xnom::Matrix,unom::Matrix,solver::String)
    N = fs.N
    ix = fs.dynamics.ix
    iu = fs.dynamics.iu
    iq = fs.funl_dynamics.iq
    is = fs.funl_dynamics.is

    Qbar = fs.solution.Q
    Sbar = fs.solution.S
    Zbar = fs.solution.Z

    Sx = fs.scaling.Sx
    iSx = fs.scaling.iSx
    Su = fs.scaling.Su
    iSu = fs.scaling.iSu

    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
    else
        println("You should select Mosek or Clarabel")
    end


    # cvx variables (scaled)
    Qcvx = []
    Scvx = []
    Zcvx = []
    for i in 1:(N+1)
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Scvx, @variable(model, [1:iu, 1:iu], PSD))
        push!(Zcvx, @variable(model, [1:ix, 1:ix]))
    end
    @variable(model, vc[1:iq,1:N])
    @variable(model, vc_t[1:N])

    # scale reference trajectory
    Qbar_scaled = zeros(ix,ix,N+1)
    Sbar_scaled = zeros(iu,iu,N+1)
    Zbar_scaled = zeros(ix,ix,N+1)

    for i in 1:N+1
        Qbar_scaled[:,:,i] .= iSx*Qbar[:,:,i]*iSx
        Sbar_scaled[:,:,i] .= iSu*Sbar[:,:,i]*iSu
        Zbar_scaled[:,:,i] .= iSx*Zbar[:,:,i]*iSx
    end

    # boundary condition
    @constraint(model, Sx*Qcvx[1]*Sx >= fs.solution.Qi, PSDCone())
    @constraint(model, Sx*Qcvx[end]*Sx <= fs.solution.Qf, PSDCone())

    # funnel dynamics
    for i in 1:N
        Qi = Sx*Qcvx[i]*Sx
        Si = Su*Scvx[i]*Su
        Zi = Sx*Zcvx[i]*Sx
        Qip = Sx*Qcvx[i+1]*Sx
        Sip = Su*Scvx[i+1]*Su
        Zip = Sx*Zcvx[i+1]*Sx
        @constraint(model, vec(Qip) == fs.solution.Aq[:,:,i]*vec(Qi) +
            fs.solution.Bm[:,:,i]*vec(Si) + fs.solution.Bp[:,:,i]*vec(Sip) +
            fs.solution.Sm[:,:,i]*vec(Zi) + fs.solution.Sp[:,:,i]*vec(Zip))
    end

    # Block LMI
    for i in 1:N+1
        @constraint(model, 0 <= Zcvx[i], PSDCone())
        # @constraint(model, 0 .== Zcvx[i])
    end
    
    # constraint
    N_constraint = size(fs.funl_constraint,1)
    for i in 1:N+1
        for j in 1:N_constraint
            A,B = diff(fs.dynamics,xnom[:,i],unom[:,i])
            Y = -0.5*(Su*Scvx[i]*Su)*B'
            impose!(fs.funl_constraint[j],model,Sx*Qcvx[i]*Sx,Y,xnom[:,i],unom[:,i])
        end
    end

    # cost
    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    cost_funl = - tr(Sx*Qcvx[1]*Sx) + tr(Sx*Qcvx[end]*Sx)
    # cost_funl = - log_det_Q

    # # penalty
    for i in 1:N
        @constraint(model, [vc_t[i]; vc[:,i]] in MOI.NormOneCone(1 + iq))
    end
    cost_vc = sum([vc_t[i] for i in 1:N])

    # trust region
    @variable(model, tr_norm[1:3,1:N+1])
    for i in 1:N+1
        @constraint(model, [tr_norm[1,i]; vec(Qcvx[i]-Qbar_scaled[:,:,i])] in SecondOrderCone())
        @constraint(model, [tr_norm[2,i]; vec(Scvx[i]-Sbar_scaled[:,:,i])] in SecondOrderCone())
        @constraint(model, [tr_norm[3,i]; vec(Zcvx[i]-Zbar_scaled[:,:,i])] in SecondOrderCone())
    end
    cost_tr = sum(tr_norm)

    cost_all = fs.w_funl * cost_funl + fs.w_vc * cost_vc + fs.w_tr * cost_tr
   
    @objective(model,Min,cost_all)
    optimize!(model)

    for i in 1:N+1
        fs.solution.Q[:,:,i] = Sx*value.(Qcvx[i])*Sx
        fs.solution.S[:,:,i] = Su*value.(Scvx[i])*Su
        fs.solution.Z[:,:,i] = Sx*value.(Zcvx[i])*Sx
    end
    return value(cost_all),value(cost_funl),value(cost_vc),value(cost_tr)
end

function run(fs::FunnelSynthesis,Q0::Array{Float64,3},S0::Array{Float64,3},Z0::Array{Float64,3},Qi::Matrix,Qf::Matrix,xnom::Matrix,unom::Matrix,dtnom::Vector,solver::String)
    fs.solution.Q .= Q0
    fs.solution.S .= S0
    fs.solution.Z .= Z0

    fs.solution.Qi .= Qi 
    fs.solution.Qf .= Qf

    for iteration in 1:fs.max_iter
        # discretization & linearization
        fs.solution.Aq,fs.solution.Bm,fs.solution.Bp,fs.solution.Sm,fs.solution.Sp = discretize_foh(fs.funl_dynamics,
            fs.dynamics,xnom[:,1:N],unom,dtnom,fs.solution.Q[:,:,1:N],fs.solution.S,fs.solution.Z)

        # solve subproblem
        c_all, c_funl, c_vc, c_tr = sdpopt!(fs,xnom,unom,"Mosek")

        # propagate
        Qfwd,fs.solution.tprop,fs.solution.xprop,fs.solution.uprop,fs.solution.Qprop,fs.solution.Sprop =  propagate_multiple_FOH(DLMI,dynamics,
            xnom,unom,dtnom,fs.solution.Q,fs.solution.S,fs.solution.Z)
        dyn_error = maximum([norm(Qfwd[:,:,i] - fs.solution.Q[:,:,i],2) for i in 1:N+1])

        if fs.verbosity == true && iteration == 1
            println("+--------------------------------------------------------------------------------------------------+")
            println("|                                   ..:: Penalized Trust Region ::..                               |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
            println("| iter. |    cost    |    tof    |   funl    |   rate    |  param  | log(vc) | log(tr)  | log(dyn) |")
            println("+-------+------------+-----------+-----------+-----------+---------+---------+----------+----------+")
        end
        @printf("|%-2d     |%-7.2f     |%-7.3f   |%-7.3f    |%-7.3f    |%-5.3f    |%-5.1f    | %-5.1f    |%-5.1e   |\n",
            iteration,
            c_all,-1,c_funl,-1,
            -1,
            log10(abs(c_vc)), log10(abs(c_tr)), log10(abs(dyn_error)))

        flag_vc::Bool = c_vc < fs.tol_vc
        flag_tr::Bool = c_tr < fs.tol_tr
        flag_dyn::Bool = dyn_error < fs.tol_dyn

        if flag_vc && flag_tr && flag_dyn
            println("+--------------------------------------------------------------------------------------------------+")
            println("Converged!")
            break
        end
    end
end