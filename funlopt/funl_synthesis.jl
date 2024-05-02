using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel

include("funl_dynamics.jl")
include("funl_utils.jl")
include("funl_constraint.jl")
include("funl_ctcs.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/scaling.jl")

mutable struct FunnelSolution
    Q::Array{Float64,3}
    K::Array{Float64,3}
    Y::Array{Float64,3}
    Z::Array{Float64,3}

    # Lsmooth
    b::Vector{Float64}
    θ::Float64

    Aq::Array{Float64,3}
    Bym::Array{Float64,3}
    Byp::Array{Float64,3}
    Bzm::Array{Float64,3}
    Bzp::Array{Float64,3}

    Qi::Matrix{Float64}
    Qf::Matrix{Float64}

    t::Vector{Float64}
    tprop::Any
    xprop::Any
    uprop::Any
    Qprop::Any
    Kprop::Any
    Yprop::Any
    Zprop::Any
    ctcs_fwd::Any


    # matrix for CTCS
    Sq::Array{Float64,3}
    Sym::Array{Float64,3}
    Syp::Array{Float64,3}
    Szm::Array{Float64,3}
    Szp::Array{Float64,3}
    Sbm::Array{Float64,3}
    Sbp::Array{Float64,3}
    SZ::Array{Float64,2}

    # bounded disturbance
    # uncertainty

    function FunnelSolution(N::Int64,ix::Int64,iu::Int64,iq::Int64,iy::Int64)
        Q = zeros(ix,ix,N+1)
        K = zeros(iu,ix,N+1)
        Y = zeros(iu,Int64(iy/iu),N+1)
        Z = zeros(ix,ix,N+1)
        b = ones(N+1)
        θ = 0.5

        Aq = zeros(iq,iq,N)
        Bym = zeros(iq,iy,N)
        Byp = zeros(iq,iy,N)
        Bzm = zeros(iq,iq,N)
        Bzp = zeros(iq,iq,N)

        Qi = zeros(ix,ix)
        Qf = zeros(ix,ix)
        
        t = zeros(N+1)
        new(Q,K,Y,Z,b,θ,Aq,Bym,Byp,Bzm,Bzp,Qi,Qf,t)
        # new()
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

    flag_type::String
    funl_ctcs::Union{FunnelCTCS,Nothing}
    function FunnelSynthesis(N::Int,max_iter::Int,
        dynamics::Dynamics,funl_dynamics::FunnelDynamics,funl_constraint::Vector{T},scaling::Scaling,
        w_funl::Float64,w_vc::Float64,w_tr::Float64,tol_tr::Float64,tol_vc::Float64,tol_dyn::Float64,
        verbosity::Bool;flag_type::String="Linear",funl_ctcs::Any=nothing) where T <: FunnelConstraint
        ix = dynamics.ix
        iu = dynamics.iu
        iq = funl_dynamics.iq
        iy = funl_dynamics.iy
        solution = FunnelSolution(N,ix,iu,iq,iy)
        if funl_ctcs === nothing
            println(flag_type," funnel and CTCS is ignored")
        else
            println(flag_type," funnel and CTCS is considered after first iteration")
        end
        new(dynamics,funl_dynamics,funl_constraint,scaling,solution,
            N,w_funl,w_vc,w_tr,tol_tr,tol_vc,tol_dyn,max_iter,verbosity,
            flag_type,funl_ctcs)
    end
end

function LMILsmooth!(fs,model::Model,Q::Matrix,Y::Matrix,Z::Matrix,b::Any,idx::Int64,iteration::Int64)
    θ = fs.solution.θ
    iψ = fs.dynamics.iψ
    iμ = fs.dynamics.iμ

    tmp12 = fs.dynamics.Cv*Q + fs.dynamics.Dvu*Y
    Bound_b = [b * I(fs.dynamics.iv) tmp12;
        tmp12' Q
    ]
    @constraint(model, 0 <= Bound_b, PSDCone())
    N11 = diagm(θ ./ ( fs.dynamics.β .* fs.dynamics.β))
    N22 =  b * θ .* Matrix{Float64}(I, iψ, iψ)
    LMI11 = -Z
    LMI21 = N22 * fs.dynamics.G'
    LMI22 = -N22
    LMI31 = fs.dynamics.Cμ * Q + fs.dynamics.Dμu * Y
    LMI32 = zeros(iμ,iψ)
    LMI33 = -N11
    LMI = [LMI11 LMI21' LMI31';
        LMI21 LMI22 LMI32';
        LMI31 LMI32 LMI33
    ]
    # iH = size(LMI,1)
    @constraint(model, LMI <= 0, PSDCone())
    # @constraint(model, LMI <= -fs.funl_ctcs.epsilon .* Matrix{Float64}(I,iH,iH), PSDCone())
end

function boundary_initial!(fs,model::Model,Q1)
    @constraint(model, Q1 >= fs.solution.Qi, PSDCone())
end

function boundary_final!(fs,model::Model,Qend)
    @constraint(model, Qend <= fs.solution.Qf, PSDCone())
end

function funnel_dynamics!(fs,model::Model,Qi,Qip,Yi,Yip,Zi,Zip,i)
    @constraint(model, vec(Qip) == fs.solution.Aq[:,:,i]*vec(Qi) +
        fs.solution.Bym[:,:,i]*vec(Yi) + fs.solution.Byp[:,:,i]*vec(Yip) +
        fs.solution.Bzm[:,:,i]*vec(Zi) + fs.solution.Bzp[:,:,i]*vec(Zip))
end

function ctcs_dynamics!(fs,model::Model,Qi,Yi,Yip,Zi,Zip,bi,bip,i)
    @constraint(model, 1e-7*fs.scaling.s_ctcs*ones(fs.funl_ctcs.is) >= fs.scaling.s_ctcs*(fs.solution.Sq[:,:,i]*vec(Qi) +
        fs.solution.Sym[:,:,i]*vec(Yi) + fs.solution.Syp[:,:,i]*vec(Yip) +
        fs.solution.Szm[:,:,i]*vec(Zi) + fs.solution.Szp[:,:,i]*vec(Zip) +
        fs.solution.Sbm[:,:,i]*[bi] + fs.solution.Sbp[:,:,i]*[bip] +
        fs.solution.SZ[:,i]) 
        )
end

function state_input_constraints!(fs,model::Model,Qi,Yi,xnom,unom)
    N_constraint = size(fs.funl_constraint,1)
    for j in 1:N_constraint
        impose!(fs.funl_constraint[j],model,Qi,Yi,xnom,unom)
    end
end

function sdpopt!(fs::FunnelSynthesis,xnom::Matrix,unom::Matrix,solver::String,iteration::Int64)
    N = fs.N
    ix = fs.dynamics.ix
    iu = fs.dynamics.iu
    iq = fs.funl_dynamics.iq
    is =  fs.funl_ctcs === nothing ? 0 : fs.funl_ctcs.is

    Sx = fs.scaling.Sx
    iSx = fs.scaling.iSx
    Su = fs.scaling.Su
    iSu = fs.scaling.iSu
    
    @assert(typeof(fs.funl_dynamics) == LinearDLMI)
    Sr = Sx
    iSr = iSx
    ir = ix

    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false) # Turn off verbosity for Mosek
    else
        println("You should select Mosek or Clarabel")
    end


    # cvx variables (scaled)
    Qcvx = []
    Ycvx = []
    Zcvx = []
    for i in 1:(N+1)
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Ycvx, @variable(model, [1:iu, 1:ir]))
        push!(Zcvx, @variable(model, [1:ix, 1:ix]))
    end
    @variable(model, bcvx[1:N+1])

    # parameter
    θ = fs.solution.θ

    # Q is PD
    very_small = 1e-5
    for i in 1:N+1
        @constraint(model, Sx*Qcvx[i]*Sx >= very_small * I(ix), PSDCone())
    end

    # scale reference trajectory
    Qbar_scaled = zeros(ix,ix,N+1)
    Ybar_scaled = zeros(iu,ir,N+1)
    Zbar_scaled = zeros(ix,ix,N+1)
    bbar_scaled = zeros(N+1)
    for i in 1:N+1
        Qbar_scaled[:,:,i] .= iSx*fs.solution.Q[:,:,i]*iSx
        Ybar_scaled[:,:,i] .= iSu*fs.solution.Y[:,:,i]*iSr
        Zbar_scaled[:,:,i] .= iSx*fs.solution.Z[:,:,i]*iSx
    end
    bbar_scaled .= fs.solution.b

    # boundary condition
    boundary_initial!(fs,model,Sx*Qcvx[1]*Sx)
    boundary_final!(fs,model,Sx*Qcvx[end]*Sx)

    for i in 1:N+1
        Qi = Sx*Qcvx[i]*Sx
        Yi = Su*Ycvx[i]*Sr
        Zi = Sx*Zcvx[i]*Sx
        bi = bcvx[i]
        if i <= N
            Qip = Sx*Qcvx[i+1]*Sx
            Yip = Su*Ycvx[i+1]*Sr
            Zip = Sx*Zcvx[i+1]*Sx
            bip = bcvx[i+1]
        end

        # Funnel dynamics
        if i <= N
            funnel_dynamics!(fs,model,Qi,Qip,Yi,Yip,Zi,Zip,i)
        end

        # CTCS
        if i <= N && fs.funl_ctcs !== nothing && iteration != 1
            ctcs_dynamics!(fs,model::Model,Qi,Yi,Yip,Zi,Zip,bi,bip,i)
        end

        # Lyapunov condition
        if fs.flag_type == "Linear"
            @constraint(model, 0 <= Zcvx[i], PSDCone())
        elseif fs.flag_type == "Lsmooth"
            LMILsmooth!(fs,model,Qi,Yi,Zi,bi,i,iteration)
        end

        # constraints
        state_input_constraints!(fs,model::Model,Qi,Yi,xnom[:,i],unom[:,i])
    end

    # cost
    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    # cost_funl = - tr(Sx*Qcvx[1]*Sx) + tr(Sx*Qcvx[end]*Sx)
    # cost_funl = tr(Sx*Qcvx[end]*Sx)
    cost_funl = - log_det_Q # not stable
    # cost_funl = - tr(Sx*Qcvx[1]*Sx)

    # virtual control
    # for i in 1:N
    #     @constraint(model, [vc_t[i]; vc[:,i]] in MOI.NormOneCone(1 + is))
    # end
    # cost_vc = sum([vc_t[i] for i in 1:N])
    cost_vc = 0

    # trust region
    cost_tr = sum([dot(vec(Qcvx[i]-Qbar_scaled[:,:,i]),vec(Qcvx[i]-Qbar_scaled[:,:,i])) +
        dot(vec(Ycvx[i]-Ybar_scaled[:,:,i]),vec(Ycvx[i]-Ybar_scaled[:,:,i])) +
        dot(vec(Zcvx[i]-Zbar_scaled[:,:,i]),vec(Zcvx[i]-Zbar_scaled[:,:,i]))
        for i in 1:N+1])
    w_tr = iteration > 1 ? fs.w_tr : 0

    cost_all = fs.w_funl * cost_funl + fs.w_vc * cost_vc + w_tr * cost_tr
   
    @objective(model,Min,cost_all)
    optimize!(model)

    for i in 1:N+1
        fs.solution.Q[:,:,i] .= Sx*value.(Qcvx[i])*Sx
        fs.solution.Y[:,:,i] .= Su*value.(Ycvx[i])*Sr
        fs.solution.Z[:,:,i] .= Sx*value.(Zcvx[i])*Sx
    end
    fs.solution.b = value.(bcvx)

    return value(cost_all),value(cost_funl),value(cost_vc),value(cost_tr)
end

function run(fs::FunnelSynthesis,Q0::Array{Float64,3},Y0::Array{Float64,3},Z0::Array{Float64,3},Qi::Matrix,Qf::Matrix,xnom::Matrix,unom::Matrix,dtnom::Vector,solver::String,θ0=nothing)
    fs.solution.Q .= Q0
    fs.solution.Y .= Y0
    fs.solution.Z .= Z0

    fs.solution.Qi .= Qi 
    fs.solution.Qf .= Qf

    if θ0 !== nothing
        fs.solution.θ = θ0
    end

    for iteration in 1:fs.max_iter
        # discretization & linearization
        if fs.funl_ctcs === nothing
            fs.solution.Aq,fs.solution.Bym,fs.solution.Byp,fs.solution.Bzm,fs.solution.Bzp = discretize_foh(fs.funl_dynamics,
                fs.dynamics,xnom[:,1:N],unom,dtnom,fs.solution.Q[:,:,1:N],fs.solution.Y,fs.solution.Z)
        else
            (
                fs.solution.Aq,fs.solution.Bym,fs.solution.Byp,fs.solution.Bzm,fs.solution.Bzp,
                fs.solution.Sq,fs.solution.Sym,fs.solution.Syp,
                fs.solution.Szm,fs.solution.Szp,
                fs.solution.Sbm,fs.solution.Sbp,
                fs.solution.SZ,
                ~,~,fs.solution.ctcs_fwd 
            ) = discretize_foh(fs.funl_dynamics,fs.dynamics,fs.funl_ctcs,
                xnom[:,1:N],unom,dtnom,fs.solution.Q[:,:,1:N],fs.solution.Y,fs.solution.Z,fs.solution.b)
            # println(s_prop)
        end

        # solve subproblem
        c_all, c_funl, c_vc, c_tr = sdpopt!(fs,xnom,unom,solver,iteration)

        # propagate
        (
            Qfwd,
            fs.solution.tprop,fs.solution.xprop,fs.solution.uprop,
            fs.solution.Qprop,fs.solution.Yprop,fs.solution.Zprop
            ) =  propagate_multiple_FOH(fs.funl_dynamics,fs.dynamics,
            xnom,unom,dtnom,fs.solution.Q,fs.solution.Y,fs.solution.Z,false)
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