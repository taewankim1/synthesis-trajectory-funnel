
include("../trajopt/dynamics.jl")
# include("../trajopt/discretize.jl")
include("funl_dynamics.jl")
using LinearAlgebra
using LaTeXStrings

function create_block_diagonal(right::Matrix, n::Int)
    blocks = [right for _ in 1:n]
    return BlockDiagonal(blocks)
end

function get_index_for_upper(n::Int)
    matrix = reshape([i for i in 1:n^2], n, n)
    return vec_upper(matrix,n)
end

function vec_upper(A::Matrix{T},m::Int64) where T
    v = Vector{T}(undef, div(m*(m+1),2))
    # v = zeros(m*(m+1))
    k = 0
    @inbounds for j in 1:m
        @inbounds for i = 1:j
            v[k + i] = A[i,j]
        end
        k += j
    end
    return v
end

function mat_upper(v::Vector{T},n::Int64) where {T}
    M = zeros(T, n, n)
    @assert size(M, 2) == n
    @assert length(v) >= n * (n + 1) / 2
    k = 0
    @inbounds for j in 1:n
        @inbounds @simd for i = 1:j
            M[i,j] = v[k+i]
            M[j,i] = v[k+i]
        end
        k += j
    end
    return M
end
# Xnom,ULnom,URnom = QYZ_to_XULR(Qnom,Ynom,ZLnom,ZRnom);
function QYZ_to_XULR(Q::Array{Float64,3},Y::Array{Float64,3},ZL::Array{Float64,3},ZR::Array{Float64,3})
    N = size(Q,3) - 1
    ix = size(Q,1)
    iu = size(Y,1)
    iq = div(ix*(ix+1),2)
    iy = ix*iu
    q = Matrix{Float64}(undef,iq,N+1)
    yl = Matrix{Float64}(undef,iy,N)
    yr = Matrix{Float64}(undef,iy,N)
    zl = Matrix{Float64}(undef,iq,N)
    zr = Matrix{Float64}(undef,iq,N)
    for i in 1:N
        q[:,i] .= vec_upper(Q[:,:,i],ix)
        yl[:,i] .= vec(Y[:,:,i])
        yr[:,i] .= vec(Y[:,:,i+1])
        zl[:,i] .= vec_upper(ZL[:,:,i],ix)
        zr[:,i] .= vec_upper(ZR[:,:,i],ix)
    end
    q[:,N+1] .= vec_upper(Q[:,:,N+1],ix)
    return q,vcat(yl,zl),vcat(yr,zr)
end

function XULR_to_QYZ(X::Matrix{Float64},UL::Matrix{Float64},UR::Matrix{Float64},ix::Int,iu::Int)
    N = size(X,2) - 1
    iq = size(X,1)
    Q = Array{Float64}(undef,ix,ix,N+1)
    Y = Array{Float64}(undef,iu,ix,N+1)
    ZL = Array{Float64}(undef,ix,ix,N)
    ZR = Array{Float64}(undef,ix,ix,N)
    for i in 1:N+1
        Q[:,:,i] .= mat_upper(X[:,i],ix) 
        if i <= N
            ZL[:,:,i] .= mat_upper(UL[ix*iu+1:ix*iu+iq,i],ix) 
            ZR[:,:,i] .= mat_upper(UR[ix*iu+1:ix*iu+iq,i],ix) 
        end
    end
    Y[:,:,1:N] .= reshape(UL[1:ix*iu,:],(iu,ix,N))
    Y[:,:,N+1] .= reshape(UR[1:ix*iu,N],(iu,ix))
    return Q,Y,ZL,ZR
end

function XU_to_QYZ(X::Matrix{Float64},U::Matrix{Float64},ix::Int,iu::Int)
    N = size(X,2)
    iq = size(X,1)
    Q = Array{Float64}(undef,ix,ix,N)
    Z = Array{Float64}(undef,ix,ix,N)
    for i in 1:N
        Q[:,:,i] .= mat_upper(X[:,i],ix) 
        Z[:,:,i] .= mat_upper(U[ix*iu+1:ix*iu+iq,i],ix) 
    end
    Y = reshape(U[1:ix*iu,:],(iu,ix,N))
    return Q,Y,Z
end

function QYZ_to_XU(Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    N = size(Q,3)
    ix = size(Q,1)
    iu = size(Y,1)
    iq = div(ix*(ix+1),2)
    q = Matrix{Float64}(undef,iq,N)
    z = Matrix{Float64}(undef,iq,N)
    for i in 1:N
        q[:,i] .= vec_upper(Q[:,:,i],ix)
        z[:,i] .= vec_upper(Z[:,:,i],ix)
    end
    y = reshape(Y,(iu*ix,N))
    return q,vcat(y,z)
end

function QYZS_to_XU(Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3},S::Matrix{Float64})
    N = size(Q,3)
    ix = size(Q,1)
    iu = size(K,1)
    iq = div(ix*(ix+1),2)
    q = Matrix{Float64}(undef,iq,N)
    z = Matrix{Float64}(undef,iq,N)
    for i in 1:N
        q[:,i] .= vec_upper(Q[:,:,i],ix)
        z[:,i] .= vec_upper(Z[:,:,i],ix)
    end
    y = reshape(Y,(iu*ix,N))
    # z = reshape(Z,(ix*ix,N))
    return q,vcat(y,z,S)
end

function get_radius_angle_Ellipse2D(Q_list)
    radius_list = []
    angle_list = []

    for i in 1:size(Q_list,3)
        Q_ = Q_list[:,:,i]
        # eigval = eigvals(inv(Q_))
        # radius = sqrt.(1 ./ eigval)
        # println("radius of x,y,theta: ", radius)
        A = [1 0 0; 0 1 0]
        Q_proj = A * Q_ * A'
        Q_inv = inv(Q_proj)
        eigval, eigvec = eigen(Q_inv)
        radius = sqrt.(1 ./ eigval)
        # println("radius of x and y: ", radius)
        rnew = eigvec * [radius[1]; 0]
        angle = atan(rnew[2], rnew[1])
        push!(radius_list, radius)
        push!(angle_list, angle)
    end
    return radius_list, angle_list
end

function propagate_multiple_FOH(model::FunnelDynamics,dynamics::Dynamics,
    x::Matrix,u::Matrix,T::Vector,
    X::Matrix,UL::Matrix,UR::Matrix;
    flag_single::Bool=false)
    N = size(x,2) - 1
    ix = model.ix
    iu = model.iu
    iX = model.iX
    iU = model.iU

    idx_x = 1:ix
    idx_X = (ix+1):(ix+iX)
    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        Um = p[3]
        Up = p[4]
        dt = p[5]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        U_ = alpha * Um + beta * Up

        x_ = V[idx_x]
        X_ = V[idx_X]

        # traj terms
        f = forward(dynamics,x_,u_)
        fx,fu = diff(dynamics,x_,u_)

        # funl terms
        F = forward(model,X_,U_,A=fx,B=fu)

        dxdt = f
        dqdt = F
        dV = [dxdt;dqdt]
        out .= dV[:]
    end

    tprop = []
    xprop = []
    uprop = []
    Xprop = []
    Uprop = []
    Xfwd = zeros(size(X))
    Xfwd[:,1] .= X[:,1]
    for i = 1:N
        if flag_single == true
            V0 = [x[:,i];Xfwd[:,i]][:]
        else
            V0 = [x[:,i];X[:,i]][:]
        end

        um = u[:,i]
        up = u[:,i+1]
        Um = UL[:,i]
        Up = UR[:,i]
        dt = T[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,Um,Up,dt))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);

        ode = stack(sol.u)
        tode = sol.t
        xode = ode[idx_x, :]
        Xode = ode[idx_X, :]

        uode = zeros(iu,size(tode,1))
        Uode = zeros(iU,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt
            uode[:,idx] .= alpha * um + beta * up
            Uode[:,idx] .= alpha * Um + beta * Up
        end

        tprop = i == 1 ? tode : vcat(tprop, sum(T[1:i-1]) .+ tode)
        xprop = i == 1 ? xode : hcat(xprop, xode)
        uprop = i == 1 ? uode : hcat(uprop, uode)
        Xprop = i == 1 ? Xode : hcat(Xprop, Xode)
        Uprop = i == 1 ? Uode : hcat(Uprop, Uode)
        Xfwd[:,i+1] = ode[idx_X,end]
    end
    return Xfwd,tprop,xprop,uprop,Xprop,Uprop
end

# function propagate_multiple_FOH(model::FunnelDynamics,dynamics::Dynamics,
#     x::Matrix,u::Matrix,T::Vector,
#     X::Matrix,UL::Matrix,UR::Matrix;
#     flag_single::Bool=false)
function propagate_from_funnel_entry(x0::Vector,model::FunnelDynamics,dynamics::Dynamics,
    xnom::Matrix,unom::Matrix,Tnom::Vector,
    X::Matrix,UL::Matrix,UR::Matrix)
    # Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    N = size(xnom,2) - 1
    ix = model.ix
    iu = model.iu
    iX = model.iX
    iU = model.iU

    idx_x = 1:ix
    idx_xnom = ix+1:2*ix
    idx_Xnom = (2*ix+1):(2*ix+iX)
    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        Um = p[3]
        Up = p[4]
        dt = p[5]

        alpha = (dt - t) / dt
        beta = t / dt

        unom_ = alpha * um + beta * up
        Unom_ = alpha * Um + beta * Up

        x_ = V[idx_x]
        xnom_ = V[idx_xnom]
        Xnom_ = V[idx_Xnom]

        Q_ = mat_upper(Xnom_,ix) 
        Y_ = reshape(Unom_[1:ix*iu],(iu,ix))
        K_ = Y_ * inv(Q_)

        u_ = unom_ + K_ * (x_ - xnom_)

        # traj terms
        f = forward(dynamics,x_,u_)
        fnom = forward(dynamics,xnom_,unom_)
        fx,fu = diff(dynamics,x_,u_)

        # funl terms
        F = forward(model,Xnom_,Unom_,A=fx,B=fu)

        dV = [f;fnom;F]
        out .= dV[:]
    end

    # things to be saved
    xfwd = zeros(size(xnom))
    xfwd[:,1] .= x0
    tprop = []
    xprop = []
    uprop = []
    for i = 1:N
        V0 = [xfwd[:,i];xnom[:,i];X[:,i]][:]
        um = unom[:,i]
        up = unom[:,i+1]
        Um = UL[:,i]
        Up = UR[:,i]
        dt = Tnom[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,Um,Up,dt))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);

        tode = sol.t
        ode = stack(sol.u)
        xode = ode[idx_x,:]
        uode = zeros(iu,size(tode,1))
        xnomode = ode[idx_xnom,:]
        Xnomode = ode[idx_Xnom,:]
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt

            x_ = xode[:,idx]
            xnom_ = xnomode[:,idx]
            Xnom_ = Xnomode[:,idx]
            unom_ = alpha * um + beta * up
            Unom_ = alpha * Um + beta * Up

            Q_ = mat_upper(Xnom_,ix) 
            Y_ = reshape(Unom_[1:ix*iu],(iu,ix))
            K_ = Y_ * inv(Q_)
            uode[:,idx] .= unom_ + K_ * (x_ - xnom_)
        end
        if i == 1
            tprop = tode
            xprop = xode
            uprop = uode
        else 
            tprop = vcat(tprop,sum(Tnom[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            uprop = hcat(uprop,uode)
        end
        xfwd[:,i+1] = xode[:,end]
    end
    return xfwd,tprop,xprop,uprop
end