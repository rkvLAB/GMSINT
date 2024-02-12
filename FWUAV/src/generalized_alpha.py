def Generalized_alpha_step(t_k, u_k, du_k, M, C, K, h, ddu_k = None, f_k = None, f_ks = None, rho_inf = 0.):
    # Generalized-alpha parameters
    #rho_inf := 1: No dissipation; 0: Asymptotic annihilation
    # h: time step

    # (Eq. 25 Chung and Hulbert 1993)
    alpha_f = (rho_inf) / (rho_inf + 1)
    alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
    
    # (Eq. 17 Chung and Hulbert 1993)
    gamma = 1 / 2 - alpha_m + alpha_f
    # (Eq. 20 Chung and Hulbert 1993)
    beta = (1 - alpha_m + alpha_f)**2 / 4

    # Flags
    if isinstance(f_k, type(None)):
        f_k = np.zeros_like(u_k)
        f_ks = np.zeros_like(u_k)

    f_k_alpha_f = (1 - alpha_f)*f_ks + alpha_f*f_k

    
    # right-hand side of Equation
    rhs_star = f_k_alpha_f - alpha_m * (M @ ddu_k) - alpha_f * (C @ du_k + K @ u_k)
    rhs = rhs_star - (1 - alpha_f) * (C @ (du_k + h * (1 - gamma) * ddu_k) + \
                                      K @ (u_k + h * du_k + h**2*((1/2 - beta) * ddu_k)))

    S = (1 - alpha_m) * M + (1 - alpha_f) * (C * gamma * h + K * beta * h **2)

    ddu_ks = np.linalg.inv(S) @ rhs
    du_ks = du_k + h * ((1 - gamma)*ddu_k + gamma * ddu_ks)
    u_ks = u_k + h * du_k + h**2*((1/2 - beta)*ddu_k + beta*ddu_ks)
    
    return u_ks, du_ks, ddu_ks
    
if __name__ == "__main__":
    
    d0 = np.array([1.])
    v0 = np.array([0.]) 
    M = np.array([[1.]])
    K = (2 * np.pi)**2 * M #np.array([[100.]])
    C = np.array([[0.]])
    
    h = 0.05
    t_array = np.arange(0, 20 + h, h)
    u_array = np.zeros((len(t_array), 1))
    du_array = np.zeros_like(u_array)
    ddu_array = np.zeros_like(u_array)
    
    # Initial conditions
    u_array[0, :] = d0
    du_array[0, :] = v0
    ddu_array[0, :] = np.linalg.inv(M) @ (-C @ v0 - K @ d0)
    
    
    for counter, _t in enumerate(t_array[:-1]):
        u_array[counter + 1, :], du_array[counter + 1, :],  ddu_array[counter + 1, :] =  \
                        Generalized_alpha_step(0., u_array[counter, :], du_array[counter, :], M, C, K, h, ddu_k = ddu_array[counter, :], rho_inf = 0.8) 
    
    print("Omega", 2*np.pi/h)
    
    fig, ax = plt.subplots()
    ax.plot(t_array, u_array)
    ax.plot(t_array, du_array)
    ax.plot(t_array, ddu_array)
