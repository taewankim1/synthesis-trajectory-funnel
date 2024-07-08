abstract type Dynamics end

mutable struct Unicycle <: Dynamics
    ix::Int
    iu::Int
    iϕ::Int
    iv::Int
    iψ::Int
    iμ::Int
    Cv::Array{Float64,2}
    Dvu::Array{Float64,2}
    G::Union{Vector,Matrix}
    Cμ::Array{Float64,2}
    Dμu::Array{Float64,2}
    β::Vector{Float64}
    function Unicycle()
        ix = 3
        iu = 2
        iϕ = 2
        iv = 2

        iψ = iϕ
        # iψ = 1
        iμ = iv
        
        # iψ = iϕ*iv
        # iμ = iψ

        Go = [1 0;0 1;0 0]
        # Go = [1;1;0]
        Cv = [0 0 1;0 0 0] 
        Dvu = [0 0;1 0]

        G = Go
        Cμ = Cv
        Dμu = Dvu

        # G = kron(Go,ones(1,iv))
        # Cμ = kron(ones(iϕ),Cv)
        # Dμu = kron(ones(iϕ),Dvu)

        β = zeros(iψ)
        new(ix,iu,iϕ,iv,iψ,iμ,Cv,Dvu,G,Cμ,Dμu,β)
    end
end

function forward(model::Unicycle, x::Vector, u::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    
    v = u[1]
    w = u[2]

    f = zeros(size(x))
    f[1] = v * cos(x3)
    f[2] = v * sin(x3)
    f[3] = w
    return f
end

function diff(model::Unicycle, x::Vector, u::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    
    v = u[1]
    w = u[2]

    fx = zeros(model.ix,model.ix)
    fx[1,1] = 0.0
    fx[1,2] = 0.0
    fx[1,3] = - v * sin(x3)
    fx[2,1] = 0.0
    fx[2,2] = 0.0
    fx[2,3] = v * cos(x3)
    fx[3,1] = 0.0
    fx[3,2] = 0.0
    fx[3,3] = 0.0
    fu = zeros(model.ix,model.iu)
    fu[1,1] = cos(x3)
    fu[1,2] = 0.0
    fu[2,1] = sin(x3)
    fu[2,2] = 0.0 
    fu[3,1] = 0.0
    fu[3,2] = 1.0
    return fx, fu
end

function diff2(model::Unicycle, x::Vector, u::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    
    u1 = u[1]
    u2 = u[2]

    fxx = zeros(model.ix,model.ix,model.ix)
    fxx[1,1,1] = 0
    fxx[1,1,2] = 0
    fxx[1,1,3] = 0
    fxx[1,2,1] = 0
    fxx[1,2,2] = 0
    fxx[1,2,3] = 0
    fxx[1,3,1] = 0
    fxx[1,3,2] = 0
    fxx[1,3,3] = -u1*cos(x3)
    fxx[2,1,1] = 0
    fxx[2,1,2] = 0
    fxx[2,1,3] = 0
    fxx[2,2,1] = 0
    fxx[2,2,2] = 0
    fxx[2,2,3] = 0
    fxx[2,3,1] = 0
    fxx[2,3,2] = 0
    fxx[2,3,3] = -u1*sin(x3)
    fxx[3,1,1] = 0
    fxx[3,1,2] = 0
    fxx[3,1,3] = 0
    fxx[3,2,1] = 0
    fxx[3,2,2] = 0
    fxx[3,2,3] = 0
    fxx[3,3,1] = 0
    fxx[3,3,2] = 0
    fxx[3,3,3] = 0
    fxu = zeros(model.ix,model.ix,model.iu)
    fxu[1,1,1] = 0
    fxu[1,1,2] = 0
    fxu[1,2,1] = 0
    fxu[1,2,2] = 0
    fxu[1,3,1] = -sin(x3)
    fxu[1,3,2] = 0
    fxu[2,1,1] = 0
    fxu[2,1,2] = 0
    fxu[2,2,1] = 0
    fxu[2,2,2] = 0
    fxu[2,3,1] = cos(x3)
    fxu[2,3,2] = 0
    fxu[3,1,1] = 0
    fxu[3,1,2] = 0
    fxu[3,2,1] = 0
    fxu[3,2,2] = 0
    fxu[3,3,1] = 0
    fxu[3,3,2] = 0
    fux = permutedims(fxu, (1, 3, 2))
    fuu = zeros(model.ix,model.iu,model.iu)
    fuu[1,1,1] = 0
    fuu[1,1,2] = 0
    fuu[1,2,1] = 0
    fuu[1,2,2] = 0
    fuu[2,1,1] = 0
    fuu[2,1,2] = 0
    fuu[2,2,1] = 0
    fuu[2,2,2] = 0
    fuu[3,1,1] = 0
    fuu[3,1,2] = 0
    fuu[3,2,1] = 0
    fuu[3,2,2] = 0
    return fxx, fxu, fux, fuu
end

mutable struct Rocket <: Dynamics
    ix::Int64
    iu::Int64

    alpha_ME::Float64
    alpha_RCS::Float64
    J_x::Float64
    J_y::Float64
    J_z::Float64

    r_t::Float64
    g::Float64
    function Rocket()
        g0 = 9.81
        I_ME = 300.0
        alpha_ME = 1 / (g0*I_ME)
        I_RCS = 200
        alpha_RCS = 1 / (g0*I_RCS)

        m0 = (1500+750)/2
        J_x = m0 * 4.2
        J_y = m0 * 4.2
        J_z = m0 * 0.6

        r_t = 1.0
        g = 1.625
        new(13,6,alpha_ME,alpha_RCS,J_x,J_y,J_z,r_t,g)
    end
end

function forward(model::Rocket,x::Vector,u::Vector)
    m = x[1]
    rx = x[2]
    ry = x[3]
    rz = x[4]
    vx = x[5]
    vy = x[6]
    vz = x[7]
    phi = x[8]
    theta = x[9]
    psi = x[10]
    p = x[11]
    q = x[12]
    r = x[13]

    Fx = u[1]
    Fy = u[2]
    Fz = u[3]
    Tx = u[4]
    Ty = u[5]
    Tz = u[6]

    alpha_m = model.alpha_ME
    alpha_r = model.alpha_RCS
    J_x = model.J_x
    J_y = model.J_y
    J_z = model.J_z

    r_t = model.r_t
    g = model.g

    f = zeros(size(x))
    f[1] = -alpha_m*sqrt(Fx^2 + Fy^2 + Fz^2) - alpha_r*sqrt(Tx^2 + Ty^2 + Tz^2)
    f[2] = vx
    f[3] = vy
    f[4] = vz
    f[5] = Fx*cos(psi)*cos(theta)/m + Fy*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))/m + Fz*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/m
    f[6] = Fx*sin(psi)*cos(theta)/m + Fy*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))/m + Fz*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))/m
    f[7] = -Fx*sin(theta)/m + Fy*sin(phi)*cos(theta)/m + Fz*cos(phi)*cos(theta)/m - g
    f[8] = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
    f[9] = q*cos(phi) - r*sin(phi)
    f[10] = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)
    f[11] = (Fy*r_t + J_y*q*r - J_z*q*r + Tx)/J_x
    f[12] = (-Fx*r_t - J_x*p*r + J_z*p*r + Ty)/J_y
    f[13] = (J_x*p*q - J_y*p*q + Tz)/J_z
    return f
end

mutable struct ThreeDOFManipulatorDynamics <: Dynamics
    ix::Int
    iu::Int
    iϕ::Int
    iv::Int
    iψ::Int
    iμ::Int
    Cv::Array{Float64,2}
    Dvu::Array{Float64,2}
    G::Union{Vector,Matrix}
    Cμ::Array{Float64,2}
    Dμu::Array{Float64,2}
    β::Vector{Float64}

    m1::Float64
    m2::Float64
    m3::Float64
    l1::Float64
    l2::Float64
    l3::Float64
    lg1::Float64
    lg2::Float64
    lg3::Float64
    I1::Float64
    I2::Float64
    I3::Float64
    g::Float64
    function ThreeDOFManipulatorDynamics(g::Float64)
        ix = 6
        iu = 3
        iϕ = 3
        iv = 3

        iψ = iϕ
        iμ = iv

        Go = [0 0 0;0 0 0;0 0 0;1 0 0;0 1 0;0 0 1]
        Cv = Matrix(1.0I,ix,ix)
        Dvu = zeros(iu,iu)

        G = Go
        Cμ = Cv
        Dμu = Dvu

        β = zeros(iψ)

        m1 = 1
        m2 = 1
        m3 = 1
        l1 = 1
        l2 = 1
        l3 = 1
        lg1 = 0.5*l1
        lg2 = 0.5*l2
        lg3 = 0.5*l3
        I1 = 1/12*m1*l1^2
        I2 = 1/12*m2*l2^2
        I3 = 1/12*m3*l3^2
        new(ix,iu,iϕ,iv,iψ,iμ,Cv,Dvu,G,Cμ,Dμu,β,
        m1,m2,m3,l1,l2,l3,lg1,lg2,lg3,I1,I2,I3,g)
    end
end

function forward(model::ThreeDOFManipulatorDynamics, x::Vector, u::Vector)
    q1 = x[1]
    q2 = x[2]
    q3 = x[3]
    dq1 = x[4]
    dq2 = x[5]
    dq3 = x[6]
    
    tau1 = u[1]
    tau2 = u[2]
    tau3 = u[3]

    m1 = model.m1
    m2 = model.m2
    m3 = model.m3
    l1 = model.l1
    l2 = model.l2
    l3 = model.l3
    l_g1 = model.lg1
    l_g2 = model.lg2
    l_g3 = model.lg3
    I1 = model.I1
    I2 = model.I2
    I3 = model.I3
    g = model.g

    f = zeros(size(x))
    f[1] = dq1
    f[2] = dq2
    f[3] = dq3
    f[4] = l1*(-dq1*l_g3*m3*(dq1*(l1*sin(q2 + q3) + l2*sin(q3)) + dq2*l2*sin(q3)) - dq2*l2*l_g3*m3*(dq1 + dq2)*sin(q3) - g*l_g3*m3*cos(q1 + q2 + q3) + tau3)*(-I2*l_g3*m3*cos(q2 + q3) + I3*l2*m3*cos(q2) + I3*l_g2*m2*cos(q2) + l2^2*l_g3*m3^2*sin(q2)*sin(q3) + l2*l_g2*l_g3*m2*m3*cos(q2)*cos(q3) + l2*l_g3^2*m3^2*cos(q2) - l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) - l_g2^2*l_g3*m2*m3*cos(q2 + q3) + l_g2*l_g3^2*m2*m3*cos(q2))/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3) + (-dq1*(dq1*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l2*l_g3*m3*sin(q3)) + dq2*dq3*l2*l_g3*m3*sin(q3) - dq3*l2*l_g3*m3*(-dq1 - dq2 - dq3)*sin(q3) - g*(l_g2*m2*cos(q1 + q2) + m3*(l2*cos(q1 + q2) + l_g3*cos(q1 + q2 + q3))) + tau2)*(-I2*I3 - I2*l_g3^2*m3 - I3*l1*l2*m3*cos(q2) - I3*l1*l_g2*m2*cos(q2) - I3*l2^2*m3 - I3*l_g2^2*m2 - l1*l2*l_g3^2*m3^2*cos(q2) + l1*l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) - l1*l_g2*l_g3^2*m2*m3*cos(q2) - l2^2*l_g3^2*m3^2*sin(q3)^2 - l_g2^2*l_g3^2*m2*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3) + (-dq1*(-dq2*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))) - dq2*(-dq1*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq2*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))*(-dq1 - dq2 - dq3) - g*(l_g1*m1*cos(q1) + m2*(l1*cos(q1) + l_g2*cos(q1 + q2)) + m3*(l1*cos(q1) + l2*cos(q1 + q2) + l_g3*cos(q1 + q2 + q3))) + tau1)*(I2*I3 + I2*l_g3^2*m3 + I3*l2^2*m3 + I3*l_g2^2*m2 + l2^2*l_g3^2*m3^2*sin(q3)^2 + l_g2^2*l_g3^2*m2*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3)
    f[5] = (-dq1*l_g3*m3*(dq1*(l1*sin(q2 + q3) + l2*sin(q3)) + dq2*l2*sin(q3)) - dq2*l2*l_g3*m3*(dq1 + dq2)*sin(q3) - g*l_g3*m3*cos(q1 + q2 + q3) + tau3)*(-I1*I3 - I1*l2*l_g3*m3*cos(q3) - I1*l_g3^2*m3 + I2*l1*l_g3*m3*cos(q2 + q3) - I3*l1^2*m2 - I3*l1^2*m3 - I3*l1*l2*m3*cos(q2) - I3*l1*l_g2*m2*cos(q2) - I3*l_g1^2*m1 - l1^2*l2*l_g3*m2*m3*cos(q3) + l1^2*l2*l_g3*m3^2*cos(q2)*cos(q2 + q3) - l1^2*l2*l_g3*m3^2*cos(q3) + l1^2*l_g2*l_g3*m2*m3*cos(q2)*cos(q2 + q3) - l1^2*l_g3^2*m2*m3 + l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 - l1^2*l_g3^2*m3^2 - l1*l2^2*l_g3*m3^2*cos(q2)*cos(q3) + l1*l2^2*l_g3*m3^2*cos(q2 + q3) - l1*l2*l_g2*l_g3*m2*m3*cos(q2)*cos(q3) - l1*l2*l_g3^2*m3^2*cos(q2) + l1*l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) + l1*l_g2^2*l_g3*m2*m3*cos(q2 + q3) - l1*l_g2*l_g3^2*m2*m3*cos(q2) - l2*l_g1^2*l_g3*m1*m3*cos(q3) - l_g1^2*l_g3^2*m1*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3) + (-dq1*(dq1*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l2*l_g3*m3*sin(q3)) + dq2*dq3*l2*l_g3*m3*sin(q3) - dq3*l2*l_g3*m3*(-dq1 - dq2 - dq3)*sin(q3) - g*(l_g2*m2*cos(q1 + q2) + m3*(l2*cos(q1 + q2) + l_g3*cos(q1 + q2 + q3))) + tau2)*(I1*I3 + I1*l_g3^2*m3 + I2*I3 + I2*l_g3^2*m3 + I3*l1^2*m2 + I3*l1^2*m3 + 2*I3*l1*l2*m3*cos(q2) + 2*I3*l1*l_g2*m2*cos(q2) + I3*l2^2*m3 + I3*l_g1^2*m1 + I3*l_g2^2*m2 + l1^2*l_g3^2*m2*m3 - l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + l1^2*l_g3^2*m3^2 + 2*l1*l2*l_g3^2*m3^2*cos(q2) - 2*l1*l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) + 2*l1*l_g2*l_g3^2*m2*m3*cos(q2) - l2^2*l_g3^2*m3^2*cos(q3)^2 + l2^2*l_g3^2*m3^2 + l_g1^2*l_g3^2*m1*m3 + l_g2^2*l_g3^2*m2*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3) + (-dq1*(-dq2*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))) - dq2*(-dq1*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq2*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))*(-dq1 - dq2 - dq3) - g*(l_g1*m1*cos(q1) + m2*(l1*cos(q1) + l_g2*cos(q1 + q2)) + m3*(l1*cos(q1) + l2*cos(q1 + q2) + l_g3*cos(q1 + q2 + q3))) + tau1)*(-I2*I3 - I2*l_g3^2*m3 - I3*l1*l2*m3*cos(q2) - I3*l1*l_g2*m2*cos(q2) - I3*l2^2*m3 - I3*l_g2^2*m2 - l1*l2*l_g3^2*m3^2*cos(q2) + l1*l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) - l1*l_g2*l_g3^2*m2*m3*cos(q2) - l2^2*l_g3^2*m3^2*sin(q3)^2 - l_g2^2*l_g3^2*m2*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3)
    f[6] = l1*(-dq1*(-dq2*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))) - dq2*(-dq1*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq2*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))) - dq3*l_g3*m3*(l1*sin(q2 + q3) + l2*sin(q3))*(-dq1 - dq2 - dq3) - g*(l_g1*m1*cos(q1) + m2*(l1*cos(q1) + l_g2*cos(q1 + q2)) + m3*(l1*cos(q1) + l2*cos(q1 + q2) + l_g3*cos(q1 + q2 + q3))) + tau1)*(-I2*l_g3*m3*cos(q2 + q3) + I3*l2*m3*cos(q2) + I3*l_g2*m2*cos(q2) + l2^2*l_g3*m3^2*sin(q2)*sin(q3) + l2*l_g2*l_g3*m2*m3*cos(q2)*cos(q3) + l2*l_g3^2*m3^2*cos(q2) - l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) - l_g2^2*l_g3*m2*m3*cos(q2 + q3) + l_g2*l_g3^2*m2*m3*cos(q2))/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3) + (-dq1*l_g3*m3*(dq1*(l1*sin(q2 + q3) + l2*sin(q3)) + dq2*l2*sin(q3)) - dq2*l2*l_g3*m3*(dq1 + dq2)*sin(q3) - g*l_g3*m3*cos(q1 + q2 + q3) + tau3)*(I1*I2 + I1*I3 + I1*l2^2*m3 + 2*I1*l2*l_g3*m3*cos(q3) + I1*l_g2^2*m2 + I1*l_g3^2*m3 + I2*l1^2*m2 + I2*l1^2*m3 + I2*l_g1^2*m1 + I3*l1^2*m2 + I3*l1^2*m3 + I3*l_g1^2*m1 + l1^2*l2^2*m2*m3 - l1^2*l2^2*m3^2*cos(q2)^2 + l1^2*l2^2*m3^2 - 2*l1^2*l2*l_g2*m2*m3*cos(q2)^2 + 2*l1^2*l2*l_g3*m2*m3*cos(q3) - 2*l1^2*l2*l_g3*m3^2*cos(q2)*cos(q2 + q3) + 2*l1^2*l2*l_g3*m3^2*cos(q3) - l1^2*l_g2^2*m2^2*cos(q2)^2 + l1^2*l_g2^2*m2^2 + l1^2*l_g2^2*m2*m3 - 2*l1^2*l_g2*l_g3*m2*m3*cos(q2)*cos(q2 + q3) + l1^2*l_g3^2*m2*m3 - l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + l1^2*l_g3^2*m3^2 + l2^2*l_g1^2*m1*m3 + 2*l2*l_g1^2*l_g3*m1*m3*cos(q3) + l_g1^2*l_g2^2*m1*m2 + l_g1^2*l_g3^2*m1*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3) + (-dq1*(dq1*l1*(l2*m3*sin(q2) + l_g2*m2*sin(q2) + l_g3*m3*sin(q2 + q3)) - dq3*l2*l_g3*m3*sin(q3)) + dq2*dq3*l2*l_g3*m3*sin(q3) - dq3*l2*l_g3*m3*(-dq1 - dq2 - dq3)*sin(q3) - g*(l_g2*m2*cos(q1 + q2) + m3*(l2*cos(q1 + q2) + l_g3*cos(q1 + q2 + q3))) + tau2)*(-I1*I3 - I1*l2*l_g3*m3*cos(q3) - I1*l_g3^2*m3 + I2*l1*l_g3*m3*cos(q2 + q3) - I3*l1^2*m2 - I3*l1^2*m3 - I3*l1*l2*m3*cos(q2) - I3*l1*l_g2*m2*cos(q2) - I3*l_g1^2*m1 - l1^2*l2*l_g3*m2*m3*cos(q3) + l1^2*l2*l_g3*m3^2*cos(q2)*cos(q2 + q3) - l1^2*l2*l_g3*m3^2*cos(q3) + l1^2*l_g2*l_g3*m2*m3*cos(q2)*cos(q2 + q3) - l1^2*l_g3^2*m2*m3 + l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 - l1^2*l_g3^2*m3^2 - l1*l2^2*l_g3*m3^2*cos(q2)*cos(q3) + l1*l2^2*l_g3*m3^2*cos(q2 + q3) - l1*l2*l_g2*l_g3*m2*m3*cos(q2)*cos(q3) - l1*l2*l_g3^2*m3^2*cos(q2) + l1*l2*l_g3^2*m3^2*cos(q3)*cos(q2 + q3) + l1*l_g2^2*l_g3*m2*m3*cos(q2 + q3) - l1*l_g2*l_g3^2*m2*m3*cos(q2) - l2*l_g1^2*l_g3*m1*m3*cos(q3) - l_g1^2*l_g3^2*m1*m3)/(I1*I2*I3 + I1*I2*l_g3^2*m3 + I1*I3*l2^2*m3 + I1*I3*l_g2^2*m2 - I1*l2^2*l_g3^2*m3^2*cos(q3)^2 + I1*l2^2*l_g3^2*m3^2 + I1*l_g2^2*l_g3^2*m2*m3 + I2*I3*l1^2*m2 + I2*I3*l1^2*m3 + I2*I3*l_g1^2*m1 + I2*l1^2*l_g3^2*m2*m3 - I2*l1^2*l_g3^2*m3^2*cos(q2 + q3)^2 + I2*l1^2*l_g3^2*m3^2 + I2*l_g1^2*l_g3^2*m1*m3 + I3*l1^2*l2^2*m2*m3 - I3*l1^2*l2^2*m3^2*cos(q2)^2 + I3*l1^2*l2^2*m3^2 - 2*I3*l1^2*l2*l_g2*m2*m3*cos(q2)^2 - I3*l1^2*l_g2^2*m2^2*cos(q2)^2 + I3*l1^2*l_g2^2*m2^2 + I3*l1^2*l_g2^2*m2*m3 + I3*l2^2*l_g1^2*m1*m3 + I3*l_g1^2*l_g2^2*m1*m2 - l1^2*l2^2*l_g3^2*m2*m3^2*cos(q3)^2 + l1^2*l2^2*l_g3^2*m2*m3^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2)^2 + 2*l1^2*l2^2*l_g3^2*m3^3*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l2^2*l_g3^2*m3^3*cos(q3)^2 - l1^2*l2^2*l_g3^2*m3^3*cos(q2 + q3)^2 + l1^2*l2^2*l_g3^2*m3^3 - 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)^2 + 2*l1^2*l2*l_g2*l_g3^2*m2*m3^2*cos(q2)*cos(q3)*cos(q2 + q3) - l1^2*l_g2^2*l_g3^2*m2^2*m3*cos(q2)^2 + l1^2*l_g2^2*l_g3^2*m2^2*m3 - l1^2*l_g2^2*l_g3^2*m2*m3^2*cos(q2 + q3)^2 + l1^2*l_g2^2*l_g3^2*m2*m3^2 - l2^2*l_g1^2*l_g3^2*m1*m3^2*cos(q3)^2 + l2^2*l_g1^2*l_g3^2*m1*m3^2 + l_g1^2*l_g2^2*l_g3^2*m1*m2*m3)
    return f
end

mutable struct QuadrotorDynamics <: Dynamics
    ix::Int
    iu::Int
    # iϕ::Int
    # iv::Int
    # iψ::Int
    # iμ::Int
    # Cv::Array{Float64,2}
    # Dvu::Array{Float64,2}
    # G::Union{Vector,Matrix}
    # Cμ::Array{Float64,2}
    # Dμu::Array{Float64,2}
    # β::Vector{Float64}

    m::Float64
    g::Float64
    Jx::Float64
    Jy::Float64
    Jz::Float64
    function QuadrotorDynamics()
        ix = 12
        iu = 4
        # iϕ = 6
        # iv = 7

        # iψ = iϕ
        # iμ = iv

        # Go = [0 0 0;0 0 0;0 0 0;1 0 0;0 1 0;0 0 1]
        # Cv = Matrix(1.0I,ix,ix)
        # Dvu = zeros(iu,iu)

        # G = Go
        # Cμ = Cv
        # Dμu = Dvu

        # β = zeros(iψ)

        m = 1
        Jx = 1
        Jy = 1
        Jz = 1
        g = 9.81
        new(ix,iu,m,g,Jx,Jy,Jz)
    end
end

function forward(model::QuadrotorDynamics, x::Vector, u::Vector)
    # rx = x[1]
    # ry = x[2]
    # rz = x[3]
    vx = x[4]
    vy = x[5]
    vz = x[6]

    phi = x[7]
    theta = x[8]
    psi = x[9]

    p = x[10]
    q = x[11]
    r = x[12]
    
    Fz = u[1]
    Mx = u[2]
    My = u[3]
    Mz = u[4]

    m = model.m
    g = model.g
    J_x = model.Jx
    J_y = model.Jy
    J_z = model.Jz

    f = zeros(size(x))
    f[1] = vx
    f[2] = vy
    f[3] = vz
    f[4] = -Fz*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/m
    f[5] = -Fz*(-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))/m
    f[6] = -Fz*cos(phi)*cos(theta)/m + g
    f[7] = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
    f[8] = q*cos(phi) - r*sin(phi)
    f[9] = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)
    f[10] = (J_y*q*r - J_z*q*r + Mx)/J_x
    f[11] = (-J_x*p*r + J_z*p*r + My)/J_y
    f[12] = (J_x*p*q - J_y*p*q + Mz)/J_z
    return f
end