include("../trajopt/dynamics.jl")
include("../funlopt/funl_dynamics.jl")
include("../trajopt/discretize.jl")
using LinearAlgebra

abstract type FunnelCTCS end

struct QPD <: FunnelCTCS
    M::Float64
    N::Float64
    epsilon::Float64
    is::Int
    ix::Int
    iu::Int

    iq::Int
    iy::Int
    function QPD(M,N,epsilon,ix,iu)
        @assert M >= 1
        @assert N >= 0
        @assert epsilon >= 0
        is = 1
        new(M,N,epsilon,is,ix,iu,ix*ix,ix*iu)
    end
end

function forward(model::QPD, q::Vector, y::Vector, z::Vector)
    # -M - logdetQ
    Q = reshape(q,(model.ix,model.ix))
    M = model.M
    N = model.N
    epsilon = model.epsilon * I(ix)
    g = log(M) - log(N*det(Q-epsilon) + M)
    return max(0,g)^2
end

function diff(model::QPD, q::Vector, y::Vector, z::Vector)
    Q = reshape(q,(model.ix,model.ix))
    Aq = zeros(model.is,model.iq)
    Bq = zeros(model.is,model.iy)
    Sq = zeros(model.is,model.iq)
    M = model.M
    N = model.N
    epsilon = model.epsilon * I(ix)

    g = log(M) - log(N*det(Q-epsilon) + M)
    if g >= 0
        dfg = 2*g
    else
        dfg = 0
    end
    dg = - N*det(Q-epsilon) / (N*det(Q-epsilon) + M) * inv(Q-epsilon)
    Aq[1,:] .= dfg*vec(dg)
    return Aq,Bq,Sq
end

function discretize_foh(model::FunnelDynamics,dynamics::Dynamics,ctcs::FunnelCTCS,
    x::Matrix,u::Matrix,T::Vector,
    Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    # Q,Y,Z are unnecessary for linear DLMI, but keep them here for writing nonlinearDLMI later
    @assert size(x,2) == size(Q,3)
    @assert size(x,2) + 1 == size(u,2)
    @assert size(u,2) == size(Y,3)
    @assert size(Y,3) == size(Z,3)

    N = size(x,2)
    ix = model.ix
    iq = model.iq
    iy = model.iy
    is = ctcs.is
    function get_interval(start,size)
        return start:(start+size-1)
    end

    idx_x = get_interval(1,ix)
    idx_q = get_interval(idx_x[end]+1,iq)
    idx_A = get_interval(idx_q[end]+1,iq*iq)
    idx_Bym = get_interval(idx_A[end]+1,iq*iy)
    idx_Byp = get_interval(idx_Bym[end]+1,iq*iy)
    idx_Bzm = get_interval(idx_Byp[end]+1,iq*iq)
    idx_Bzp = get_interval(idx_Bzm[end]+1,iq*iq)
    
    idx_s = get_interval(idx_Bzp[end]+1,is)
    idx_Sq = get_interval(idx_s[end]+1,is*iq)
    idx_Sym = get_interval(idx_Sq[end]+1,is*iy)
    idx_Syp = get_interval(idx_Sym[end]+1,is*iy)
    idx_Szm = get_interval(idx_Syp[end]+1,is*iq)
    idx_Szp = get_interval(idx_Szm[end]+1,is*iq)

    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        ym = p[3]
        yp = p[4]
        zm = p[5]
        zp = p[6]
        dt = p[7]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        y_ = alpha * ym + beta * yp
        z_ = alpha * zm + beta * zp

        x_ = V[idx_x]
        q_ = V[idx_q]
        Phi = reshape(V[idx_A], (iq, iq))
        Bym_ = reshape(V[idx_Bym],(iq,iy))
        Byp_ = reshape(V[idx_Byp],(iq,iy))
        Bzm_ = reshape(V[idx_Bzm],(iq,iq))
        Bzp_ = reshape(V[idx_Bzp],(iq,iq))

        s_ = V[idx_s]
        Sq_ = reshape(V[idx_Sq], (is, iq))
        Sym_ = reshape(V[idx_Sym],(is,iy))
        Syp_ = reshape(V[idx_Syp],(is,iy))
        Szm_ = reshape(V[idx_Szm],(is,iq))
        Szp_ = reshape(V[idx_Szp],(is,iq))

        # traj terms
        f = forward(dynamics,x_,u_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)
        dAq,dBy,dBz = diff(model,A,B)
        # ctcs terms
        ds = forward(ctcs,q_,y_,z_)
        dSq,dSy,dSz = diff(ctcs,q_,y_,z_)
        # print_jl(dSq)
        # print_jl(Bym_)
        # print_jl(dSy)

        dxdt = f
        dqdt = F
        dAdt = dAq*Phi
        dBymdt = dAq*Bym_ + dBy.*alpha
        dBypdt = dAq*Byp_ + dBy.*beta
        dBzmdt = dAq*Bzm_ + dBz.*alpha
        dBzpdt = dAq*Bzp_ + dBz.*beta
        dsdt = ds
        dSqdt = dSq*Phi
        dSymdt = dSq*Bym_ + dSy.*alpha
        dSypdt = dSq*Byp_ + dSy.*beta
        dSzmdt = dSq*Bzm_ + dSz.*alpha
        dSzpdt = dSq*Bzp_ + dSz.*beta


        dV = [dxdt;dqdt;
            dAdt[:];dBymdt[:];dBypdt[:];dBzmdt[:];dBzpdt[:];
            dsdt;dSqdt[:];dSymdt[:];dSypdt[:];dSzmdt[:];dSzpdt[:]]
        out .= dV[:]
    end

    Aq = zeros(iq,iq,N)
    Bym = zeros(iq,iy,N)
    Byp = zeros(iq,iy,N)
    Bzm = zeros(iq,iq,N)
    Bzp = zeros(iq,iq,N)

    Sq = zeros(is,iq,N)
    Sym = zeros(is,iy,N)
    Syp = zeros(is,iy,N)
    Szm = zeros(is,iq,N)
    Szp = zeros(is,iq,N)
    SZ = zeros(is,N)

    x_prop = zeros(ix,N)
    q_prop = zeros(iq,N)
    s_prop = zeros(is,N)

    for i = 1:N
        A0 = I(iq)
        Bym0 = zeros(iq,iy)
        Byp0 = zeros(iq,iy)
        Bzm0 = zeros(iq,iq)
        Bzp0 = zeros(iq,iq)
        S0 = zeros(is,iq)
        Sym0 = zeros(is,iy)
        Syp0 = zeros(is,iy)
        Szm0 = zeros(is,iq)
        Szp0 = zeros(is,iq)
        V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bym0[:];Byp0[:];Bzm0[:];Bzp0[:];
            zeros(is);S0[:];Sym0[:];Syp0[:];Szm0[:];Szp0[:]][:]

        um = u[:,i]
        up = u[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        zm = vec(Z[:,:,i])
        zp = vec(Z[:,:,i+1])
        dt = T[i]

        t, sol = RK4(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt),50)
        x_prop[:,i] .= sol[idx_x,end]
        q_prop[:,i] .= sol[idx_q,end]
        Aq[:,:,i] .= reshape(sol[idx_A,end],iq,iq)
        Bym[:,:,i] .= reshape(sol[idx_Bym,end],iq,iy)
        Byp[:,:,i] .= reshape(sol[idx_Byp,end],iq,iy)
        Bzm[:,:,i] .= reshape(sol[idx_Bzm,end],iq,iq)
        Bzp[:,:,i] .= reshape(sol[idx_Bzp,end],iq,iq)
        s_prop[:,i] .= sol[idx_s,end]
        Sq[:,:,i] .= reshape(sol[idx_Sq,end],is,iq)
        Sym[:,:,i] .= reshape(sol[idx_Sym,end],is,iy)
        Syp[:,:,i] .= reshape(sol[idx_Syp,end],is,iy)
        Szm[:,:,i] .= reshape(sol[idx_Szm,end],is,iq)
        Szp[:,:,i] .= reshape(sol[idx_Szp,end],is,iq)
        SZ[:,i] .= (s_prop[:,i] - Sq[:,:,i]*vec(Q[:,:,i]) 
            - Sym[:,:,i]*ym - Syp[:,:,i]*yp
            - Szm[:,:,i]*zm - Szp[:,:,i]*zp)
    end
    return Aq,Bym,Byp,Bzm,Bzp,Sq,Sym,Syp,Szm,Szp,SZ,x_prop,q_prop,s_prop
end