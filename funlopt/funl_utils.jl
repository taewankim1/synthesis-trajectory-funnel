using LinearAlgebra

function get_radius_angle_Ellipse2D(Q_list)
    radius_list = []
    angle_list = []

    for i in 1:size(Q_list,3)
        Q_ = Q_list[:,:,i]
        eigval = eigvals(inv(Q_))
        radius = sqrt.(1 ./ eigval)
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


# def get_radius_angle(Q_list) :
#     radius_list = []
#     angle_list = []
#     for Q_ in Q_list :
#         eig,_ = np.linalg.eig(np.linalg.inv(Q_))
#         radius = np.sqrt(1/eig)
#         # print("radius of x,y,theta",radius)
#         A = np.array([[1,0,0],[0,1,0]])
#         # Q_proj = project_ellipse(Q_) 
#         Q_proj = A@Q_@A.T
#         Q_inv = np.linalg.inv(Q_proj)
#         eig,eig_vec = np.linalg.eig(Q_inv)
#         radius = np.sqrt(1/eig)
#         # print("radius of x and y",radius)
#         rnew = eig_vec@np.array([[radius[0]],[0]])
#         angle = np.arctan2(rnew[1],rnew[0])
#         radius_list.append(radius)
#         angle_list.append(angle)

#     return radius_list,angle_list