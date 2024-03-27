include("funl_utils.jl")
include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")
using LinearAlgebra
abstract type FunnelDynamics end

function com_mat(m, n)
    A = reshape(1:m*n, n, m)  # Note the swapped dimensions for Julia
    v = reshape(A', :)

    P = Matrix{Int}(I, m*n, m*n)  # Create identity matrix
    P = P[v, :]
    return P'
end

struct LinearDLMI <: FunnelDynamics
    alpha::Float64 # decay rate
    ix::Int
    iu::Int

    iq::Int
    iy::Int

    Cm::Matrix # commutation matrix
    function LinearDLMI(alpha,ix,iu)
        Cm = com_mat(iu,ix)
        new(alpha,ix,iu,ix*ix,ix*iu,Cm)
    end
end

function forward(model::LinearDLMI, q::Vector, y::Vector, z::Vector, A::Matrix, B::Matrix)
    Q = reshape(q,(model.ix,model.ix))
    Y = reshape(y,(model.iu,model.ix))
    Z = reshape(z,(model.ix,model.ix))
   
    L = A*Q + B*Y
    dQ = L + L' + model.alpha*Q + Z
    return vec(dQ)
end

function diff(model::LinearDLMI,A::Matrix,B::Matrix)
    Imat = I(model.ix)
    Aq = kron(Imat,A) + kron(A,Imat) + model.alpha * kron(Imat,Imat)
    Bq = kron(Imat,B) + kron(B,Imat) * model.Cm
    Sq = kron(Imat,Imat)
    return Aq,Bq,Sq
end

function discretize_foh(model::LinearDLMI,dynamics::Dynamics,
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

    idx_x = 1:ix
    idx_q = (ix+1):(ix+iq)
    idx_A = (ix+iq+1):(ix+iq+iq*iq)
    idx_Bm = (ix+iq+iq*iq+1):(ix+iq+iq*iq+iq*iy)
    idx_Bp = (ix+iq+iq*iq+iq*iy+1):(ix+iq+iq*iq+iq*iy+iq*iy)
    idx_Sm = (ix+iq+iq*iq+iq*iy+iq*iy+1):(ix+iq+iq*iq+iq*iy+iq*iy+iq*iq)
    idx_Sp = (ix+iq+iq*iq+iq*iy+iq*iy+iq*iq+1):(ix+iq+iq*iq+iq*iy+iq*iy+iq*iq+iq*iq)

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
        Bm_ = reshape(V[idx_Bm],(iq,iy))
        Bp_ = reshape(V[idx_Bp],(iq,iy))
        Sm_ = reshape(V[idx_Sm],(iq,iq))
        Sp_ = reshape(V[idx_Sp],(iq,iq))

        # traj terms
        f = forward(dynamics,x_,u_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)
        Aq_,Bq,Sq = diff(model,A,B)

        dxdt = f
        dqdt = F
        dAdt = Aq_*Phi
        dBmdt = Aq_*Bm_ + Bq.*alpha
        dBpdt = Aq_*Bp_ + Bq.*beta
        dSmdt = Aq_*Sm_ + Sq.*alpha
        dSpdt = Aq_*Sp_ + Sq.*beta
        dV = [dxdt;dqdt;dAdt[:];dBmdt[:];dBpdt[:];dSmdt[:];dSpdt[:]]
        out .= dV[:]
    end

    Aq = zeros(iq,iq,N)
    Bm = zeros(iq,iy,N)
    Bp = zeros(iq,iy,N)
    Sm = zeros(iq,iq,N)
    Sp = zeros(iq,iq,N)

    x_prop = zeros(ix,N)
    q_prop = zeros(iq,N)

    for i = 1:N
        A0 = I(iq)
        Bm0 = zeros(iq,iy)
        Bp0 = zeros(iq,iy)
        Sm0 = zeros(iq,iq)
        Sp0 = zeros(iq,iq)
        V0 = [x[:,i];vec(Q[:,:,i]);A0[:];Bm0[:];Bp0[:];Sm0[:];Sp0[:]][:]

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
        Bm[:,:,i] .= reshape(sol[idx_Bm,end],iq,iy)
        Bp[:,:,i] .= reshape(sol[idx_Bp,end],iq,iy)
        Sm[:,:,i] .= reshape(sol[idx_Sm,end],iq,iq)
        Sp[:,:,i] .= reshape(sol[idx_Sp,end],iq,iq)
    end
    return Aq,Bm,Bp,Sm,Sp,x_prop,q_prop
end

function propagate_multiple_FOH(model::LinearDLMI,dynamics::Dynamics,
        x::Matrix,u::Matrix,T::Vector,
        Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    N = size(x,2) - 1
    ix = model.ix
    iu = model.iu
    iq = model.iq
    iy = model.iy

    idx_x = 1:ix
    idx_q = (ix+1):(ix+iq)

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

        # traj terms
        f = forward(dynamics,x_,u_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)

        dxdt = f
        dqdt = F
        dV = [dxdt;dqdt]
        out .= dV[:]
    end

    tprop = []
    xprop = []
    uprop = []
    qprop = []
    yprop = []
    Qfwd = zeros(size(Q))
    Qfwd[:,:,1] = Q[:,:,1]
    for i = 1:N
        V0 = [x[:,i];vec(Q[:,:,i])][:]

        um = u[:,i]
        up = u[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        zm = vec(Z[:,:,i])
        zp = vec(Z[:,:,i+1])
        dt = T[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt))
        sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6;verbose=false);

        tode = sol.t
        uode = zeros(iu,size(tode,1))
        yode = zeros(iu*ix,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt
            uode[:,idx] .= alpha * um + beta * up
            yode[:,idx] .= alpha * ym + beta * yp
        end
        ode = stack(sol.u)
        xode = ode[idx_x,:]
        qode = ode[idx_q,:]
        if i == 1
            tprop = tode
            xprop = xode
            uprop = uode
            qprop = qode
            yprop = yode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            uprop = hcat(uprop,uode)
            qprop = hcat(qprop,qode)
            yprop = hcat(yprop,yode)
        end
        Qfwd[:,:,i+1] = reshape(qode[:,end],(ix,ix))
    end
    Qprop = zeros(ix,ix,length(tprop))
    Yprop = zeros(iu,ix,length(tprop))
    for i in 1:length(tprop)
        Qprop[:,:,i] .= reshape(qprop[:,i],(ix,ix))
        Yprop[:,:,i] .= reshape(yprop[:,i],(iu,ix))
    end
    return Qfwd,tprop,xprop,uprop,Qprop,Yprop
end