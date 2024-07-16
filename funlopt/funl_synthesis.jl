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

    function FunnelSolution(N::Int64,ix::Int64,iu::Int64,iq::Int64,iy::Int64,iz::Int64)
        Q = zeros(ix,ix,N+1)
        K = zeros(iu,ix,N+1)
        Y = zeros(iu,Int64(iy/iu),N+1)
        Z = zeros(iz,ix,N+1)
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
        if typeof(funl_dynamics) == LinearSOH
            iz = 2*ix
        else
            iz = ix
        end
        solution = FunnelSolution(N,ix,iu,iq,iy,iz)
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

function get_block_LMI(fs,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
    if typeof(fs.funl_dynamics) == LinearFOH || typeof(fs.funl_dynamics) == LinearSOH
        Ai,Bi = diff(fs.dynamics,xi,ui)
        Aj,Bj = diff(fs.dynamics,xj,uj)
        Wij = Ai*Qj + Qj'*Ai' + Bi*Yj + Yj'*Bi' + 0.5*fs.funl_dynamics.alpha*(Qj+Qj') - Z
        Wji = Aj*Qi + Qi'*Aj' + Bj*Yi + Yi'*Bj' + 0.5*fs.funl_dynamics.alpha*(Qi+Qi') - Z
    elseif typeof(fs.funl_dynamics) == LinearDLMI
        Wij = - Z
        Wji = - Z
    end
    LMI11 = Wij + Wji
    if fs.flag_type == "Linear"
        return LMI11
    end
    θ = fs.solution.θ
    iμ = fs.dynamics.iμ
    iψ = fs.dynamics.iψ
    N11 = diagm(θ ./ ( fs.dynamics.β .* fs.dynamics.β))
    N22i =  bi * θ .* Matrix{Float64}(I, iψ, iψ)
    N22j =  bj * θ .* Matrix{Float64}(I, iψ, iψ)
    LMI21 = (N22i+N22j) * fs.dynamics.G'
    LMI22 = -(N22i+N22j)
    LMI31i = fs.dynamics.Cμ * Qi + fs.dynamics.Dμu * Yi
    LMI31j = fs.dynamics.Cμ * Qj + fs.dynamics.Dμu * Yj
    LMI31 = LMI31i + LMI31j
    LMI32 = 2*zeros(iμ,iψ)
    LMI33 = -2*N11
    LMI = 0.5 * [LMI11 LMI21' LMI31';
        LMI21 LMI22 LMI32';
        LMI31 LMI32 LMI33
    ]
    return LMI
end

function get_b_LMI(fs,Q,Y,b)
    tmp12 = fs.dynamics.Cv*Q + fs.dynamics.Dvu*Y
    Bound_b = [b * I(fs.dynamics.iv) tmp12;
        tmp12' Q
    ]
    return Bound_b
end

function block_LMIs!(fs,model::Model,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
    LMI = get_block_LMI(fs,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
    @constraint(model, LMI <= 0, PSDCone())
    return LMI
end

function bound_on_b!(fs,model::Model,Q,Y,b)
    if fs.flag_type != "Linear"
        Bound_b = get_b_LMI(fs,Q,Y,b)
        @constraint(model, 0 <= Bound_b, PSDCone())
    end
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
    @constraint(model, 1e-7 .* fs.scaling.s_ctcs*ones(fs.funl_ctcs.is) >= fs.scaling.s_ctcs*(fs.solution.Sq[:,:,i]*vec(Qi) +
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
        if typeof(fs.funl_dynamics) == LinearDLMI
            push!(Zcvx, @variable(model, [1:ix, 1:ix], PSD))
        elseif typeof(fs.funl_dynamics) == LinearSOH
            Z1 = @variable(model, [1:ix,1:ix],Symmetric)
            Z2 = @variable(model, [1:ix,1:ix],Symmetric)
            push!(Zcvx,[Z1;Z2])
        elseif typeof(fs.funl_dynamics) == LinearFOH
            push!(Zcvx, @variable(model, [1:ix, 1:ix], Symmetric))
        end
    end
    @variable(model, bcvx[1:N+1])

    # constraint for linearSOH
    if typeof(fs.funl_dynamics) == LinearFOH
        @constraint(model, Zcvx[N] .== Zcvx[N+1])
    elseif typeof(fs.funl_dynamics) == LinearSOH
        @constraint(model, Zcvx[N][ix+1:2*ix,:] .== Zcvx[N+1][1:ix,:])
        @constraint(model, Zcvx[N][ix+1:2*ix,:] .== Zcvx[N+1][ix+1:2*ix,:])
        for i in 1:N
            @constraint(model, Zcvx[i][1:ix,:] >= Zcvx[i][ix+1:2*ix,:], PSDCone())
            # @constraint(model, Zcvx[i] >= Zpcvx[i], PSDCone())
        end
    end

    # parameter
    θ = fs.solution.θ

    # Q is PD
    very_small = 1e-4
    for i in 1:N+1
        @constraint(model, Sx*Qcvx[i]*Sx >= very_small .* Matrix(1.0I,ix,ix), PSDCone())
    end

    # scale reference trajectory
    Qbar_scaled = zeros(ix,ix,N+1)
    Ybar_scaled = zeros(iu,ir,N+1)
    if typeof(fs.funl_dynamics) == LinearSOH
        Zbar_scaled = zeros(2*ix,ix,N+1)
    else
        Zbar_scaled = zeros(ix,ix,N+1)
    end
    bbar_scaled = zeros(N+1)

    for i in 1:N+1
        Qbar_scaled[:,:,i] .= iSx*fs.solution.Q[:,:,i]*iSx
        Ybar_scaled[:,:,i] .= iSu*fs.solution.Y[:,:,i]*iSr
        if typeof(fs.funl_dynamics) == LinearSOH
            Zbar_scaled[1:ix,:,i] .= iSx*fs.solution.Z[1:ix,:,i]*iSx
            Zbar_scaled[ix+1:2*ix,:,i] .= iSx*fs.solution.Z[ix+1:2*ix,:,i]*iSx
        else
            Zbar_scaled[:,:,i] .= iSx*fs.solution.Z[:,:,i]*iSx
        end
    end
    bbar_scaled .= fs.solution.b

    # boundary condition
    boundary_initial!(fs,model,Sx*Qcvx[1]*Sx)
    boundary_final!(fs,model,Sx*Qcvx[end]*Sx)

    for i in 1:N+1
        Qi = Sx*Qcvx[i]*Sx
        Yi = Su*Ycvx[i]*Sr
        Zi = Sx*Zcvx[i][1:ix,:]*Sx
        bi = bcvx[i]
        xi = xnom[:,i]
        ui = unom[:,i]
        if i <= N
            Qip = Sx*Qcvx[i+1]*Sx
            Yip = Su*Ycvx[i+1]*Sr
            if typeof(fs.funl_dynamics) == LinearSOH
                Zip = Sx*Zcvx[i][ix+1:2*ix,:]*Sx
            else
                Zip = Sx*Zcvx[i+1]*Sx
            end
            bip = bcvx[i+1]
            xip = xnom[:,i+1]
            uip = unom[:,i+1]
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
        if typeof(fs.funl_dynamics) == LinearFOH
            bound_on_b!(fs,model,Qi,Yi,bi)
            block_LMIs!(fs,model::Model,Qi,Qi,Yi,Yi,Zi,bi,bi,xi,ui,xi,ui) # pointwise
            if i <= N
                # block_LMIs!(fs,model::Model,Qi,Qip,Yi,Yip,Zi,bi,bip,xi,ui,xip,uip)
                block_LMIs!(fs,model::Model,Qip,Qip,Yip,Yip,Zi,bip,bip,xip,uip,xip,uip)
            end
        elseif typeof(fs.funl_dynamics) == LinearSOH
            bound_on_b!(fs,model,Qi,Yi,bi)
            block_LMIs!(fs,model::Model,Qi,Qi,Yi,Yi,Zi,bi,bi,xi,ui,xi,ui) # pointwise
            if i <= N
                block_LMIs!(fs,model::Model,Qip,Qip,Yip,Yip,Zip,bip,bip,xip,uip,xip,uip)
            end
        elseif typeof(fs.funl_dynamics) == LinearDLMI
            bound_on_b!(fs,model,Qi,Yi,bi)
            block_LMIs!(fs,model::Model,Qi,Qi,Yi,Yi,Zi,bi,bi,xi,ui,xi,ui) # pointwise
        else
            error("choose appropriate funnel dynamics")
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
        if typeof(fs.funl_dynamics) == LinearSOH
            fs.solution.Z[1:ix,:,i] .= Sx*value.(Zcvx[i][1:ix,:])*Sx
            fs.solution.Z[ix+1:2*ix,:,i] .= Sx*value.(Zcvx[i][ix+1:2*ix,:])*Sx
        else
            fs.solution.Z[:,:,i] .= Sx*value.(Zcvx[i])*Sx
        end
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