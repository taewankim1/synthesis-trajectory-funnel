abstract type Dynamics end

mutable struct Unicycle <: Dynamics
    ix::Int
    iu::Int
    # iw::Int
    # iq::Int
    # ip::Int
    # C::Array{Float64,2}
    # D::Array{Float64,2}
    # E::Array{Float64,2}
    # G::Array{Float64,2}
    function Unicycle()
        new(3,2)
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

