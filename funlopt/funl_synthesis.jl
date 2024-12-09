using LinearAlgebra
using Printf
using JuMP
using MosekTools
using Clarabel

# include("funl_dynamics.jl")
# include("funl_utils.jl")
# include("funl_constraint.jl")
# include("funl_ctcs.jl")
# include("../trajopt/dynamics.jl")
# include("../trajopt/scaling.jl")

mutable struct FunnelSolution
    # Q::Array{Float64,3}
    # K::Array{Float64,3}
    # Y::Array{Float64,3}
    # Z::Array{Float64,3}
    X::Matrix{Float64}
    UL::Matrix{Float64}
    UR::Matrix{Float64}
    # TODO - lambda lam::Matrix{Float64}

    # # Lsmooth
    # b::Vector{Float64}
    # θ::Float64

    A::Array{Float64,3}
    Bm::Array{Float64,3}
    Bp::Array{Float64,3}
    rem::Matrix{Float64}

    Qi::Matrix{Float64}
    Qf::Matrix{Float64}

    t::Vector{Float64}
    tprop::Any
    xprop::Any
    uprop::Any
    Xprop::Any
    Uprop::Any

    A_sub::Array{Float64,4}
    Bm_sub::Array{Float64,4}
    Bp_sub::Array{Float64,4}
    rem_sub::Array{Float64,3}
    x_sub::Array{Float64,3}
    # ctcs_fwd::Any


    # # matrix for CTCS
    # Sq::Array{Float64,3}
    # Sym::Array{Float64,3}
    # Syp::Array{Float64,3}
    # Szm::Array{Float64,3}
    # Szp::Array{Float64,3}
    # Sbm::Array{Float64,3}
    # Sbp::Array{Float64,3}
    # SZ::Array{Float64,2}
    # function FunnelSolution(N::Int64,ix::Int64,iu::Int64,iq::Int64,iy::Int64,iz::Int64)
    function FunnelSolution(N::Int64,ix::Int64,iu::Int64,ilam::Int64)
        # b = ones(N+1)
        # θ = 0.5

        iq = div(ix*(ix+1),2)
        iX = iq
        X = zeros(iX,N+1)
        iy = ix*iu
        iz = iq
        iU = iy+iz+ilam
        UL = zeros(iU,N)
        UR = zeros(iU,N)

        A = zeros(iX,iX,N)
        Bm = zeros(iX,iU,N)
        Bp = zeros(iX,iU,N)
        # Bt = zeros(iX,N)
        rem = zeros(iX,N)

        Qi = zeros(ix,ix)
        Qf = zeros(ix,ix)
        
        t = zeros(N+1)
        new(X,UL,UR,A,Bm,Bp,rem,Qi,Qf,t)
    end
end

struct FunnelSynthesis
    dynamics::Dynamics
    funl_dynamics::FunnelDynamics
    funl_constraint::Vector{FunnelConstraint}
    scaling::Any
    solution::FunnelSolution

    N::Int64  # number of subintervals (number of node - 1)
    Nsub::Int64 # number of subsubintervals for ctcs
    w_funl::Float64  # weight for funnel cost
    w_vc::Float64  # weight for virtual control
    w_tr::Float64  # weight for trust-region
    tol_tr::Float64  # tolerance for trust-region
    tol_vc::Float64  # tolerance for virtual control
    tol_dyn::Float64  # tolerance for dynamics error
    max_iter::Int64  # maximum iteration
    verbosity::Bool

    # flag_type::String
    # funl_ctcs::Union{FunnelCTCS,Nothing}
    function FunnelSynthesis(param::Dict,
        dynamics::Dynamics,funl_dynamics::FunnelDynamics,funl_constraint::Vector{T},scaling::Scaling) where T <: FunnelConstraint
        N = param["N"]
        Nsub = param["Nsub"] # number of subsubintervals for ctcs
        w_funl = param["w_funl"]  # weight for funnel cost
        w_vc = param["w_vc"]  # weight for virtual control
        w_tr = param["w_tr"]  # weight for trust-region
        tol_tr = param["tol_tr"]  # tolerance for trust-region
        tol_vc = param["tol_vc"]  # tolerance for virtual control
        tol_dyn = param["tol_dyn"]  # tolerance for dynamics error
        max_iter = param["max_iter"]  # maximum iteration
        verbosity = param["verbosity"]

        ix = dynamics.ix
        iu = dynamics.iu
        ilam = DLMI.ilam
        solution = FunnelSolution(N,ix,iu,ilam)

        new(dynamics,funl_dynamics,funl_constraint,scaling,solution,
            N,Nsub,w_funl,w_vc,w_tr,tol_tr,tol_vc,tol_dyn,max_iter,verbosity)
    end
end

# function get_block_LMI(fs,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
#     if typeof(fs.funl_dynamics) == LinearFOH || typeof(fs.funl_dynamics) == LinearSOH
#         Ai,Bi = diff(fs.dynamics,xi,ui)
#         Aj,Bj = diff(fs.dynamics,xj,uj)
#         Wij = Ai*Qj + Qj'*Ai' + Bi*Yj + Yj'*Bi' + 0.5*fs.funl_dynamics.alpha*(Qj+Qj') - Z
#         Wji = Aj*Qi + Qi'*Aj' + Bj*Yi + Yi'*Bj' + 0.5*fs.funl_dynamics.alpha*(Qi+Qi') - Z
#     elseif typeof(fs.funl_dynamics) == LinearDLMI
#         Wij = - Z
#         Wji = - Z
#     end
#     LMI11 = Wij + Wji
#     if fs.flag_type == "Linear"
#         return LMI11
#     end
#     θ = fs.solution.θ
#     iμ = fs.dynamics.iμ
#     iψ = fs.dynamics.iψ
#     N11 = diagm(θ ./ ( fs.dynamics.β .* fs.dynamics.β))
#     N22i =  bi * θ .* Matrix{Float64}(I, iψ, iψ)
#     N22j =  bj * θ .* Matrix{Float64}(I, iψ, iψ)
#     LMI21 = (N22i+N22j) * fs.dynamics.G'
#     LMI22 = -(N22i+N22j)
#     LMI31i = fs.dynamics.Cμ * Qi + fs.dynamics.Dμu * Yi
#     LMI31j = fs.dynamics.Cμ * Qj + fs.dynamics.Dμu * Yj
#     LMI31 = LMI31i + LMI31j
#     LMI32 = 2*zeros(iμ,iψ)
#     LMI33 = -2*N11
#     LMI = 0.5 * [LMI11 LMI21' LMI31';
#         LMI21 LMI22 LMI32';
#         LMI31 LMI32 LMI33
#     ]
#     return LMI
# end

# function get_b_LMI(fs,Q,Y,b)
#     tmp12 = fs.dynamics.Cv*Q + fs.dynamics.Dvu*Y
#     Bound_b = [b * I(fs.dynamics.iv) tmp12;
#         tmp12' Q
#     ]
#     return Bound_b
# end

# function block_LMIs!(fs,model::Model,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
#     LMI = get_block_LMI(fs,Qi,Qj,Yi,Yj,Z,bi,bj,xi,ui,xj,uj)
#     @constraint(model, LMI <= 0, PSDCone())
#     return LMI
# end

# function bound_on_b!(fs,model::Model,Q,Y,b)
#     if fs.flag_type != "Linear"
#         Bound_b = get_b_LMI(fs,Q,Y,b)
#         @constraint(model, 0 <= Bound_b, PSDCone())
#     end
# end

function boundary_initial!(fs,model::Model,Q1)
    @constraint(model, Q1 >= fs.solution.Qi, PSDCone())
end

function boundary_final!(fs,model::Model,Qend)
    @constraint(model, Qend <= fs.solution.Qf, PSDCone())
end

function funnel_dynamics!(fs,model::Model,Qi,Qj,ULi,URi,i)
    @constraint(model, vec_upper(Qj,ix) == (
        fs.solution.A[:,:,i]*vec_upper(Qi,ix) 
        + fs.solution.Bm[:,:,i]*ULi
        + fs.solution.Bp[:,:,i]*URi
        + fs.solution.rem[:,i]
        # + VC[:,i]
    ))
end

# function ctcs_dynamics!(fs,model::Model,Qi,Yi,Yip,Zi,Zip,bi,bip,i)
#     @constraint(model, 1e-7 .* fs.scaling.s_ctcs*ones(fs.funl_ctcs.is) >= fs.scaling.s_ctcs*(fs.solution.Sq[:,:,i]*vec(Qi) +
#         fs.solution.Sym[:,:,i]*vec(Yi) + fs.solution.Syp[:,:,i]*vec(Yip) +
#         fs.solution.Szm[:,:,i]*vec(Zi) + fs.solution.Szp[:,:,i]*vec(Zip) +
#         fs.solution.Sbm[:,:,i]*[bi] + fs.solution.Sbp[:,:,i]*[bip] +
#         fs.solution.SZ[:,i]) 
#         )
# end

function invariance_condition(fs,Qi,Yi,Zi,xi,ui)
    if type_nonlinearity == "Linear"
        if type_funnel_dynamics == "Lyapunov"
            LMI = Zi
        elseif type_funnel_dynamics == "Basic"
            Ai,Bi = diff(fs.dynamics,xi,ui)
            G = Ai*Qi + Qi'*Ai' + Bi*Yi + Yi'*Bi' + 0.5*fs.funl_dynamics.alpha*(Qi+Qi') - Zi
            LMI = - G
        end
    elseif type_nonlinearity == "Lsmooth"
        1;
    end
    return LMI
end

function state_input_constraints!(fs,model::Model,Qi,Yi,xnom,unom)
    N_constraint = size(fs.funl_constraint,1)
    for j in 1:N_constraint
        # TODO: do you think it is a good idea to use function overloading?
        @constraint(model, impose(fs.funl_constraint[j],Qi,Yi,xnom,unom) >= 0, PSDCone())
    end
end

function sdpopt!(fs::FunnelSynthesis,xnom::Matrix,unom::Matrix,dtnom::Vector,solver::String,iteration::Int64)
    N = fs.N
    Nsub = fs.Nsub
    ix = fs.dynamics.ix
    iu = fs.dynamics.iu
    iq = fs.funl_dynamics.iq
    ilam = fs.funl_dynamics.ilam

    Sx = fs.scaling.Sx
    iSx = fs.scaling.iSx
    Su = fs.scaling.Su
    iSu = fs.scaling.iSu
    if solver == "Mosek"
        model = Model(Mosek.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_LOG", fs.verbosity) # Turn off verbosity for Mosek
    elseif solver == "Clarabel"
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", fs.verbosity) # Turn off verbosity for Mosek
    else
        println("You should select Mosek or Clarabel")
    end


    # cvx variables (scaled)
    Qcvx = []
    Ycvx = []
    ZLcvx = []
    ZRcvx = []
    # lamLcvx = []
    # lamRcvx = []
    # VC = []
    for i in 1:(N+1)
        push!(Qcvx, @variable(model, [1:ix, 1:ix], PSD))
        push!(Ycvx, @variable(model, [1:iu, 1:ix]))
        if i < N+1
            if type_Z_parameterization == "ZOH"
                ZL = @variable(model, [1:ix,1:ix],Symmetric)
                ZR = ZL
            elseif type_Z_parameterization == "FOH"
                ZL = @variable(model, [1:ix,1:ix],Symmetric)
                ZR = @variable(model, [1:ix,1:ix],Symmetric)
            end
            push!(ZLcvx,ZL)
            push!(ZRcvx,ZR)
        end
    end

    # constraint for linearSOH
    if type_Z_parameterization == "FOH"
        for i in 1:N
            @constraint(model, ZLcvx[i] >= ZRcvx[i], PSDCone())
        end
    end

    # # parameter
    # θ = fs.solution.θ

    # scale reference trajectory for trust region computation
    Qbar_scaled = zeros(ix,ix,N+1)
    Ybar_scaled = zeros(iu,ix,N+1)
    ZLbar_scaled = zeros(ix,ix,N)
    ZRbar_scaled = zeros(ix,ix,N)
    if ilam > 0
        lamLbar_scaled = zeros(ilam,N)
        lamRbar_scaled = zeros(ilam,N)
    end
    if ilam == 0
        Qbar,Ybar,ZLbar,ZRbar = XULR_to_QYZ(fs.solution.X,fs.solution.UL,fs.solution.UR,ix,iu)
    # else
    end

    for i in 1:N+1
        Qbar_scaled[:,:,i] .= iSx*Qbar[:,:,i]*iSx
        if i <= N
            Ybar_scaled[:,:,i] .= iSu*Ybar[:,:,i]*iSx
            ZLbar_scaled[:,:,i] .= iSx*ZLbar[:,:,i]*iSx
            ZRbar_scaled[:,:,i] .= iSx*ZRbar[:,:,i]*iSx
        end
    end

    # bbar_scaled .= fs.solution.b

    # boundary condition
    boundary_initial!(fs,model,Sx*Qcvx[1]*Sx)
    boundary_final!(fs,model,Sx*Qcvx[end]*Sx)

    # constraints
    for i in 1:N+1
        Qi = Sx*Qcvx[i]*Sx
        Yi = Su*Ycvx[i]*Sx
        xi = xnom[:,i]
        ui = unom[:,i]
        if i <= N
            # j = i + 1
            Qj = Sx*Qcvx[i+1]*Sx
            Yj = Su*Ycvx[i+1]*Sx
            ZLi = Sx*ZLcvx[i]*Sx
            ZRi = Sx*ZRcvx[i]*Sx
            xj = xnom[:,i+1]
            uj = unom[:,i+1]
        end

        # # Funnel dynamics
        if i <= N
            ULi = vcat(vec(Yi),vec_upper(ZLi,ix))
            URi = vcat(vec(Yj),vec_upper(ZRi,ix))
            funnel_dynamics!(fs,model,Qi,Qj,ULi,URi,i)
        end

        # Q is PD
        very_small = 1e-6
        @constraint(model, Sx*Qcvx[i]*Sx >= very_small .* Matrix(1.0I,ix,ix), PSDCone())

        # Invariance condition
        if i <= N
            LMIi = invariance_condition(fs,Qi,Yi,ZLi,xi,ui)
            @constraint(model, LMIi >= 0, PSDCone())

            LMIj = invariance_condition(fs,Qj,Yj,ZRi,xj,uj)
            @constraint(model, LMIj >= 0, PSDCone())
        end

        # constraints
        state_input_constraints!(fs,model,Qi,Yi,xi,ui)

        # oversampling for ctcs
        if i <= N && Nsub > 1 
            tsub = range(0.0, stop = dtnom[i], length = Nsub+1)[2:end-1]
            dt = dtnom[i]
            for k in 1:Nsub-1 # We use 'k' instead of 'j' to avoid confusion with variables named "var" + "j" (e.g., Qj).
                tk = tsub[k]
                alpha = (dt - tk) / dt
                beta = tk / dt
                xk = fs.solution.x_sub[:,i,k]
                uk = alpha * ui + beta * uj
                Qk = mat_upper(fs.solution.A_sub[:,:,i,k]*vec_upper(Qi,ix) 
                        + fs.solution.Bm_sub[:,:,i,k]*ULi
                        + fs.solution.Bp_sub[:,:,i,k]*URi
                        + fs.solution.rem_sub[:,i,k], ix)
                Yk = alpha * Yi + beta * Yj
                Zk = alpha * ZLi + beta * ZRi

                # Q PD
                if type_funnel_dynamics == "Lyapunov"
                    @constraint(model, Qk >= very_small .* Matrix(1.0I,ix,ix), PSDCone())
                end

                # invariance
                LMIk = invariance_condition(fs,Qk,Yk,Zk,xk,uk)
                @constraint(model, LMIk >= 0, PSDCone())

                # constraints
                state_input_constraints!(fs,model,Qk,Yk,xk,uk)
            end
        end
    end

    # cost
    @variable(model, log_det_Q)
    @constraint(model, [log_det_Q; 1; vec(Sx*Qcvx[1]*Sx)] in MOI.LogDetConeSquare(ix))
    # cost_funl = - tr(Sx*Qcvx[1]*Sx) + tr(Sx*Qcvx[end]*Sx)
    # cost_funl = tr(Sx*Qcvx[end]*Sx)
    cost_funl = - log_det_Q
    # cost_funl = - tr(Sx*Qcvx[1]*Sx)

    # virtual control
    # for i in 1:N
    #     @constraint(model, [vc_t[i]; vc[:,i]] in MOI.NormOneCone(1 + is))
    # end
    # cost_vc = sum([vc_t[i] for i in 1:N])
    cost_vc = 0

    # trust region
    cost_tr = 0.0
    for i in 1:N+1
        Qdiff = vec_upper(Qcvx[i]-Qbar_scaled[:,:,i],ix) 
        cost_tr += dot(Qdiff,Qdiff)
        Ydiff = vec(Ycvx[i]-Ybar_scaled[:,:,i]) 
        cost_tr += dot(Ydiff,Ydiff)
        if i <= N
            ZLdiff = vec_upper(ZLcvx[i]-ZLbar_scaled[:,:,i],ix) 
            cost_tr += 0.5*dot(ZLdiff,ZLdiff)
            ZRdiff = vec_upper(ZRcvx[i]-ZRbar_scaled[:,:,i],ix) 
            cost_tr += 0.5*dot(ZRdiff,ZRdiff)
        end
    end
    w_tr = iteration > 1 ? fs.w_tr : 0

    cost_all = fs.w_funl * cost_funl + fs.w_vc * cost_vc + w_tr * cost_tr
   
    @objective(model,Min,cost_all)
    optimize!(model)
    time_solve = MOI.get(model, MOI.SolveTimeSec())

    for i in 1:N+1
        Q = Sx*value.(Qcvx[i]/2 + Qcvx[i]'/2)*Sx
        fs.solution.X[:,i] .= vec_upper(Q,ix)
        if i <= N
            YL = Su*value.(Ycvx[i])*Sx
            YR = Su*value.(Ycvx[i+1])*Sx

            ZL = Sx*value.(ZLcvx[i]/2 + ZLcvx[i]'/2)*Sx
            ZR = Sx*value.(ZRcvx[i]/2 + ZRcvx[i]'/2)*Sx

            fs.solution.UL[:,i] .= vcat(vec(YL),vec_upper(ZL,ix))
            fs.solution.UR[:,i] .= vcat(vec(YR),vec_upper(ZR,ix))
        end
    end
    # fs.solution.b = value.(bcvx)

    return time_solve,value(cost_all),value(cost_funl),value(cost_vc),value(cost_tr)
end

function run(fs::FunnelSynthesis,
        X0::Matrix{Float64},UL0::Matrix{Float64},UR0::Matrix{Float64},
        Qi::Matrix,Qf::Matrix,xnom::Matrix,unom::Matrix,dtnom::Vector,solver::String,θ0=nothing)

    N = fs.N
    Nsub = fs.Nsub

    fs.solution.X .= X0
    fs.solution.UL .= UL0
    fs.solution.UR .= UR0

    fs.solution.Qi .= Qi 
    fs.solution.Qf .= Qf

    # if θ0 !== nothing
    #     fs.solution.θ = θ0
    # end

    time_dict = Dict()
    for iteration in 1:fs.max_iter
        # discretization & linearization
        time_dict["time_discretization"] = @elapsed begin
            (
                fs.solution.A,fs.solution.Bm,fs.solution.Bp,_,fs.solution.rem,_,_
            ) = discretize_foh(fs.funl_dynamics,
                fs.dynamics,xnom[:,1:N],unom,dtnom,fs.solution.X[:,1:N],fs.solution.UL,fs.solution.UR)
        end

        time_dict["time_discretization_sub"] = @elapsed begin
            if Nsub > 1
                (
                    fs.solution.A_sub,fs.solution.Bm_sub,fs.solution.Bp_sub,_,fs.solution.rem_sub,fs.solution.x_sub
                ) = discretize_foh_with_Nsub(Nsub,fs.funl_dynamics,
                    fs.dynamics,xnom[:,1:N],unom,dtnom,fs.solution.X[:,1:N],fs.solution.UL,fs.solution.UR)
            end
        end

        # solve subproblem
        time_dict["time_cvxopt"] = @elapsed begin
            time_solve,c_all, c_funl, c_vc, c_tr = sdpopt!(fs,xnom,unom,dtnom,solver,iteration)
        end
        time_dict["time_solve"] = time_solve

        # propagate
        time_dict["time_multiple_shooting"] = @elapsed begin
            (
                Xfwd,
                fs.solution.tprop,fs.solution.xprop,fs.solution.uprop,
                fs.solution.Xprop,fs.solution.Uprop
            ) =  propagate_multiple_FOH(fs.funl_dynamics,fs.dynamics,
                xnom,unom,dtnom,fs.solution.X,fs.solution.UL,fs.solution.UR,flag_single=false)
        end
        dyn_error = maximum(norm.(eachcol(Xfwd - fs.solution.X), 2))

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
    return time_dict
end