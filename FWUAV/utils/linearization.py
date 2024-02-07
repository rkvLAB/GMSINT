import numpy as np 

def linearizedDynamics(fun, q, dq, ddq, *args, epsilon=1e-8, **kwargs):
    # Linearização do sistema de equações dinâmica
    # Chamada da função
    # fun(q, dq, ddq, *args, **kwargs)

    n_variables = len(q)
    dim_image = len(fun(q, dq, ddq, *args, **kwargs))

    M = np.zeros((dim_image, n_variables))
    C = np.zeros((dim_image, n_variables))
    K = np.zeros((dim_image, n_variables))

    for i in range(n_variables):
        q_pos = q.copy()
        q_neg = q.copy()

        dq_pos = q.copy()
        dq_neg = q.copy()

        ddq_pos = q.copy()
        ddq_neg = q.copy()

        q_pos[i] = q[i] + epsilon
        q_neg[i] = q[i] - epsilon

        dq_pos[i] = dq[i] + epsilon
        dq_neg[i] = dq[i] - epsilon

        ddq_pos[i] = ddq[i] + epsilon
        ddq_neg[i] = ddq[i] - epsilon

        M[:, i] = ((fun(q, dq, ddq_pos, *args, **kwargs) - fun(q, dq, ddq_neg, *args, **kwargs)) / (
                    2 * epsilon)).flatten()
        C[:, i] = ((fun(q, dq_pos, ddq, *args, **kwargs) - fun(q, dq_neg, ddq, *args, **kwargs)) / (
                    2 * epsilon)).flatten()
        K[:, i] = ((fun(q_pos, dq, ddq, *args, **kwargs) - fun(q_neg, dq, ddq, *args, **kwargs)) / (
                    2 * epsilon)).flatten()

    return M, C, K
def Jacobian(fun, x, *args, epsilon=1e-8, **kwargs):
    # TODO colocar um procedimento para verificar convergência
    # eval_Jacobian(fun, x, epsilon = 1e-8, rtol = 1e-5)
    # variar um pouco o valor de epsilon, caso a diferença fique dentro de rtol, manter, senão, usar epsilon menor
    n_variables = len(x)
    dim_image = len(fun(x, *args, **kwargs))
    Jacobian = np.zeros((dim_image, n_variables))
    for i in range(n_variables):
        x0_pos = x.copy()
        x0_neg = x.copy()

        x0_pos[i] = x[i] + epsilon
        x0_neg[i] = x[i] - epsilon

        Jacobian[:, i] = (fun(x0_pos, *args, **kwargs) - fun(x0_neg, *args, **kwargs)) / (2 * epsilon)

    return Jacobian
