import numpy as np


from qutip import *

import warnings









# A, B in {0,1,2} aka will do measurement on A system and calculate discord between A and B

def correlations(density_matrix, d0, d1, d2, A, B, base=2):

    d = [d0,d1,d2]

    dA = d[A]

    dB = d[B]

    if isinstance(density_matrix, Qobj):

        pass # do nothing since it's already a qutip object

    else:

        rho_ABC = Qobj(density_matrix, dims=[[d0, d1, d2], [d0, d1, d2]])

    if A<B:

        # tracing out the element that we don't need
        rho_AB = rho_ABC.ptrace((A,B))
        # as A<B, after the partial trace, A=0 [[at position zero]] and B = 1 now
        quantum_mutual_info = entropy_mutual(rho_AB, 0, 1, base=base)
        classical_info = classical_correlations(rho_AB, dA,dB,A_first=True)

    if B<A:

        rho_BA = rho_ABC.ptrace((B,A))
        quantum_mutual_info = entropy_mutual(rho_BA, 0, 1, base=base)
        classical_info = classical_correlations(rho_BA, dA, dB, A_first = False)

    quantum_discord = quantum_mutual_info - classical_info

    return quantum_mutual_info, classical_info, quantum_discord


def classical_correlations(rho_AB, dA,dB, A_first):

    return C_C_random_search_A(rho_AB, dA, dB, A_first)


def C_C_random_search_A(rho_AB, dA, dB, A_first, num_search_points = 1200):

    max_classical_info = 0

    for i in range(num_search_points):

        random_U = rand_unitary(dA)

        if A_first ==True:

            temp_value = C_C_projective_A(random_U, rho_AB, dA, dB)

        else:

            temp_value = C_C_projective_B(random_U, rho_AB, dA, dB)

        if temp_value > max_classical_info:

            max_classical_info = temp_value

    return max_classical_info


def C_C_projective_A(U_A,rho_AB, dA, dB,base=2):

    #first, constructing the basis measurements

    Mbasis = [ tensor(ket2dm(basis(dA,i)), qeye(dB)) for i in range(0,dA)]

    #converting things to matrices
    U = tensor(U_A, qeye(dB))
    U_matrix = U.full()
    U_dag = tensor(U_A.dag(), qeye(dB))
    U_matrix_dagger =  U_dag.full()
    Mbasis_matrices = [Mbasis[i].full() for i in range(0,dA)]
    rho_AB_matrix = rho_AB.full()

    #projective operators
    Projection_matrices = [U_matrix @ Mbasis_matrices[i] @ U_matrix_dagger for i in range (0,dA)]

    #apply each measurement
    post_measurement_matrix = [Projection_matrices[i] @ rho_AB_matrix @ Projection_matrices[i] for i in range(0,dA)]

    post_measurement_unnormalised = [Qobj(post_measurement_matrix[i],dims=[[dA,dB],[dA,dB]]) for i in range(0,dA)]

    probabilities = [np.real(np.trace(post_measurement_matrix[i])) for i in range(0, dA)]

    post_measurement_normalised = [ Qobj(np.zeros((dA*dB,dA*dB)),dims=[[dA,dB],[dA,dB]]) for i in range(0, dA)]


    for i in range(0,dA):

        if np.isclose(probabilities[i],0): #

            #then matrix is basically zero
            pass

        else:
            post_measurement_normalised[i] = post_measurement_unnormalised[i]/probabilities[i]



    conditional_B_states = [post_measurement_normalised[i].ptrace(1) for i in range(0,dA)]

    post_measurement_B_state = 0

    for i in range(0,dA):

        post_measurement_B_state+= probabilities[i]*conditional_B_states[i]

        #print(probabilities[i])

        #print(conditional_B_states[i])

    #print(post_measurement_B_state)

    first_term = entropy_vn(post_measurement_B_state,base=base)

    second_term = 0

    for i in range(0,dA):

        second_term+= probabilities[i] * entropy_vn(conditional_B_states[i], base=base)

    return first_term-second_term


def C_C_projective_B(U_A,rho_BA, dA, dB, base=2):

    #first, constructing the basis measurements
    Mbasis = [ tensor( qeye(dB), ket2dm(basis(dA,i))) for i in range(0,dA)]

    #converting things to matrices
    U = tensor( qeye(dB),U_A)
    U_matrix = U.full()
    U_dag = tensor(qeye(dB),U_A.dag())
    U_matrix_dagger =  U_dag.full()
    Mbasis_matrices = [Mbasis[i].full() for i in range(0,dA)]
    rho_BA_matrix = rho_BA.full()

    #projective operators
    Projection_matrices = [U_matrix @ Mbasis_matrices[i] @ U_matrix_dagger for i in range (0,dA)]

    #apply each measurement
    post_measurement_matrix = [Projection_matrices[i] @ rho_BA_matrix @ Projection_matrices[i] for i in range(0,dA)]
    post_measurement_unnormalised = [Qobj(post_measurement_matrix[i],dims=[[dB,dA],[dB,dA]]) for i in range(0,dA)]

    #post_measurement_normalised = [post_measurement_unnormalised[i].unit() for i in range(0, dA)]

    probabilities = [np.real(np.trace(post_measurement_matrix[i])) for i in range(0,dA)]

    post_measurement_normalised = [Qobj(np.zeros((dA * dB, dA * dB)), dims=[[dA, dB], [dA, dB]]) for i in range(0, dA)]

    for i in range(0, dA):

        if np.isclose(probabilities[i], 0):  #

            # then matrix is basically zero

            pass

        else:

            post_measurement_normalised[i] = post_measurement_unnormalised[i]/probabilities[i]

    conditional_B_states = [post_measurement_normalised [i].ptrace(0) for i in range(0,dA)]
    post_measurement_B_state = 0

    for i in range(0,dA):

        post_measurement_B_state+= probabilities[i]*conditional_B_states[i]

    first_term = entropy_vn(post_measurement_B_state,base=base)
    second_term = 0

    for i in range(0,dA):

        second_term+= probabilities[i] * entropy_vn(conditional_B_states[i], base=base)

    return first_term-second_term