import numpy as np
import scipy.special
import itertools

# Auxiliary functions.
def gaussian(r, R, alpha):
    """
    Returns a Gaussian-type function with Gaussian orbital exponent
    alpha, centered on R.

    Args:
        r (ndarray): Vector for electron coordinate.
        R (ndarray): Vector for nucleus coordinate.
        alpha (float): Gaussian orbital exponent.
    """
    r_R = r - R
    r_R2 = np.dot(r_R, r_R)
        
    return (2 * alpha / np.pi)**(0.75) * np.exp(-alpha * r_R2)

def gaussian_S(alpha, beta, RA, RB):
    """
    Computes the overlap between Gaussian-type functions with exponents
    alpha and beta, centered on nuclei A and B.
    """
    normalization = (4 * alpha * beta / (np.pi)**2)**(0.75)
    exp_coeff = (alpha * beta) / (alpha + beta)
    
    RA_RB = np.linalg.norm(RA - RB)
    RA_RB2 = RA_RB**2
    
    coeff = (np.pi / (alpha + beta))**(1.5)
    return normalization * coeff * np.exp(-exp_coeff * RA_RB2)

def gaussian_T(alpha, beta, RA, RB):
    normalization = (4 * alpha * beta / (np.pi)**2)**(0.75)
    exp_coeff = (alpha * beta) / (alpha + beta)
    
    RA_RB = np.linalg.norm(RA - RB)
    RA_RB2 = RA_RB**2
    
    coeff = exp_coeff * (3 - 2 * exp_coeff * RA_RB2) * (
            np.pi / (alpha + beta))**(1.5)
    return normalization * coeff * np.exp(-exp_coeff * RA_RB2)

def gaussian_Vnuc(alpha, beta, RA, RB, RC, ZC):
    normalization = (4 * alpha * beta / (np.pi)**2)**(0.75)
    exp_coeff = (alpha * beta) / (alpha + beta)
    erf_coeff = np.sqrt(alpha + beta)
    
    RA_RB = np.linalg.norm(RA - RB)
    RA_RB2 = RA_RB**2
    
    RP = (alpha * RA + beta * RB) / (alpha + beta)
    RP_RC = np.linalg.norm(RP - RC)
    
    if RP_RC == 0:
        coeff = -2. * np.pi * ZC / (alpha + beta)
        return normalization * coeff * np.exp(-exp_coeff * RA_RB2) 

    coeff = (- ZC / RP_RC) * (np.pi / (alpha + beta))**(1.5)
    return (normalization * coeff * np.exp(-exp_coeff * RA_RB2) 
            * scipy.special.erf(erf_coeff * RP_RC))

def gaussian_2e(alpha, beta, gamma, delta, RA, RB, RC, RD):
    normalization = (16. * alpha * beta * gamma * delta / (np.pi)**4)**(0.75)
    exp_coeff1 = (alpha * beta) / (alpha + beta)
    exp_coeff2 = (gamma * delta) / (gamma + delta)
    erf_coeff = np.sqrt(
            (alpha + beta) * (gamma + delta) / (alpha + beta + gamma + delta))
    
    RA_RB = np.linalg.norm(RA - RB)
    RA_RB2 = RA_RB**2
    RC_RD = np.linalg.norm(RC - RD)
    RC_RD2 = RC_RD**2
    
    RP = (alpha * RA + beta * RB) / (alpha + beta)
    RQ = (gamma * RC + delta * RD) / (gamma + delta)
    RP_RQ = np.linalg.norm(RP - RQ)

    if RP_RQ == 0:
        coeff = 2. * np.pi**(2.5) / ((alpha + beta) * (gamma + delta) 
                * np.sqrt(alpha + beta + gamma + delta))
        return (normalization * coeff * 
                np.exp(-exp_coeff1 * RA_RB2 - exp_coeff2 * RC_RD2))
    
    coeff = (np.pi**3 / RP_RQ) * (1. / ((alpha + beta) * (gamma + delta)))**(1.5) 
    return (normalization * coeff 
            * np.exp(-exp_coeff1 * RA_RB2 - exp_coeff2 * RC_RD2) 
            * scipy.special.erf(erf_coeff * RP_RQ))
        

class SCF:
    """
    Runs the SCF procedure to get the restricted closed shell Hartree-Fock 
    ground state of a diatomic molecule. Uses the STO-3G basis set.
    """
    def __init__(self, geometry, exponents, n_electrons):
        """
        Initialize SCF routine.

        Args:
            geometry (list(tuple)): List of (atomic number, coordinate) pairs.
            exponent (ndarray): List of alpha exponents in the STO-3G
                                basis functions.
        """
        self.geometry = geometry
        self.n_electrons = n_electrons
        self.STO_3G_coeffs = np.array([0.444635, 0.535328, 0.154329])
        self.STO_3G_exponents = exponents # matrix

        self.S = np.zeros((2, 2))
        self.S_diag = np.zeros((2, 2))
        self.U = np.zeros((2, 2))
        self.X = np.zeros((2, 2))
        self.T = np.zeros((2, 2))
        self.Vnuc = np.zeros((2, 2))
        self.Hcore = np.zeros((2, 2))
        self.P = np.zeros((2, 2))
        self.G = np.zeros((2, 2))
        self.F = np.zeros((2, 2))
        self.F_orthog = np.zeros((2, 2))

        self.C = np.zeros((2, 2))
        self.orbital_energies = np.zeros(2)
        self.Etot = 0.
        
    def get_S(self):
        """
        Compute the overlap matrix S.
        """

        def get_S_elements(mu, nu):
            """
            Computes the matrix elements of the overlap matrix S.
            """
            sum = 0.

            for i in range(3):
                for j in range(3):
                    coeff = self.STO_3G_coeffs[i] * self.STO_3G_coeffs[j]
                    alpha = self.STO_3G_exponents[mu][i]
                    beta = self.STO_3G_exponents[nu][j]
                    RA = self.geometry[mu][1]
                    RB = self.geometry[nu][1]
                    integral = gaussian_S(alpha, beta, RA, RB)
                    sum += coeff * integral

            return sum
        
        for mu in range(2):
            for nu in range(2):
                self.S[mu][nu] = get_S_elements(mu, nu)
    
    def get_T(self):
        """
        Compute the kinetic energy matrix T.
        """

        def get_T_elements(mu, nu):
            """
            Computes the matrix elements of the kinetic energy matrix T.
            """
            sum = 0.

            for i in range(3):
                for j in range(3):
                    coeff = self.STO_3G_coeffs[i] * self.STO_3G_coeffs[j]
                    alpha = self.STO_3G_exponents[mu][i]
                    beta = self.STO_3G_exponents[nu][j]
                    RA = self.geometry[mu][1]
                    RB = self.geometry[nu][1]
                    integral = gaussian_T(alpha, beta, RA, RB)
                    sum += coeff * integral

            return sum
        
        for mu in range(2):
            for nu in range(2):
                self.T[mu][nu] = get_T_elements(mu, nu)

    def get_Vnuc_C(self, C):
        """
        Compute the nuclear potential energy matrix Vnuc_C for nucleus C.
        """
        Vnuc_C = np.zeros((2, 2))

        def get_Vnuc_elements(mu, nu):
            """
            Computes the matrix elements of the nuclear potential energy 
            matrix Vnuc.
            """
            sum = 0.

            for i in range(3):
                for j in range(3):
                    coeff = self.STO_3G_coeffs[i] * self.STO_3G_coeffs[j]
                    alpha = self.STO_3G_exponents[mu][i]
                    beta = self.STO_3G_exponents[nu][j]
                    RA = self.geometry[mu][1]
                    RB = self.geometry[nu][1]
                    RC = self.geometry[C][1]
                    ZC = self.geometry[C][0]
                    integral = gaussian_Vnuc(alpha, beta, RA, RB, RC, ZC)
                    sum += coeff * integral

            return sum
        
        for mu in range(2):
            for nu in range(2):
                Vnuc_C[mu][nu] += get_Vnuc_elements(mu, nu)
        
        return Vnuc_C
    
    def get_Vnuc(self):
        """
        Compute the nuclear potential energy matrix Vnuc over all nuclei.
        """
        for C in range(2):
            self.Vnuc += self.get_Vnuc_C(C)

    def get_Hcore(self):
        """
        Compute the core Hamiltonian matrix Hcore.
        """
        # If T and Vnuc matrices are empty, compute them first.
        if np.array_equal(self.T, np.zeros((2, 2))):
                self.get_T()

        if np.array_equal(self.Vnuc, np.zeros((2, 2))):
                self.get_Vnuc()

        self.Hcore = self.T + self.Vnuc

    def get_G(self):
        """
        Compute the 2 electron matrix G.
        """

        def get_G_elements_terms(mu, nu, sigma, lambd):
            """
            Computes the terms of matrix elements of the 2 electron matrix G.
            """
            sum = 0.

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            coeff = (self.P[lambd][sigma] * self.STO_3G_coeffs[i] 
                                    * self.STO_3G_coeffs[j]
                                    * self.STO_3G_coeffs[k] 
                                    * self.STO_3G_coeffs[l])
                            alpha = self.STO_3G_exponents[mu][i]
                            beta = self.STO_3G_exponents[nu][j]
                            gamma = self.STO_3G_exponents[sigma][k]
                            delta = self.STO_3G_exponents[lambd][l]
                            RA = self.geometry[mu][1]
                            RB = self.geometry[nu][1]
                            RC = self.geometry[sigma][1]
                            RD = self.geometry[lambd][1]
                            integral1 = gaussian_2e(alpha, beta, gamma, delta,
                                                    RA, RB, RC, RD)
                            integral2 = gaussian_2e(alpha, delta, gamma, beta,
                                                    RA, RD, RC, RB)
                            sum += coeff * (integral1 - 0.5 * integral2)
            
            return sum
        
        def get_G_elements(mu, nu):
            """
            Computes the matrix elements of the 2 electron matrix G.
            """
            sum = 0.

            for lambd in range(2):
                for sigma in range(2):
                    sum += get_G_elements_terms(mu, nu, sigma, lambd)

            return sum
        
        for mu in range(2):
            for nu in range(2):
                self.G[mu][nu] = get_G_elements(mu, nu)
        
    def get_F(self):
        """
        Compute the Fock matrix F.
        """
        # If Hcore and G matrices are empty, compute them first.
        if np.array_equal(self.Hcore, np.zeros((2, 2))):
                self.get_Hcore()

        if np.array_equal(self.G, np.zeros((2, 2))):
                self.get_G()

        self.F = self.Hcore + self.G

    def diagonalize_S(self):
        eigval, self.U = np.linalg.eigh(self.S)
        self.S_diag = np.diag(eigval)        

    def get_X_symmetric_orthog(self):
        """
        Compute the symmetric orthogonalization matrix X = S^(-1/2).
        """
        diag_elements = 1. / np.diag(np.sqrt(self.S_diag))
        factor = np.matmul(np.diag(diag_elements), np.matrix.getH(self.U))
        self.X = np.matmul(self.U, factor)

    def get_F_orthog(self):
        """
        Compute the Fock matrix in the new basis.
        """
        factor = np.matmul(self.F, self.X)
        X_adjoint = np.matrix.getH(self.X)
        self.F_orthog = np.matmul(X_adjoint, factor)

    def get_C_and_e(self):
        """
        Returns the coefficient matrix C without updating the internal 
        C matrix.
        """
        eigval, eigvec = np.linalg.eigh(self.F_orthog)
        return (eigval, np.matmul(self.X, eigvec))

    def get_P(self, C):
        """
        Compute the density matrix P.
        Returns the density  matrix P without updating the internal 
        P matrix.
        """
        P = np.zeros((2, 2))

        def get_P_elements(mu, nu):
            """
            Computes the matrix elements of the density matrix P.
            """
            sum = 0.

            for a in range(int(self.n_electrons / 2)):
                sum += C[mu][a] * np.conjugate(C)[nu][a]

            return 2. * sum

        for mu in range(2):
            for nu in range(2):
                P[mu][nu] = get_P_elements(mu, nu)

        return P

    def get_P_error(self, P1, P2):
        """
        Computes the standard deviation of successive density matrix elements
        between P1 and P2.
        """
        coeff = 1. / 3**2 
        Pdiff = P1 - P2 
        Pdiff2 = np.power(Pdiff, 2)
        return np.sqrt(coeff * np.sum(Pdiff2))
    
    def get_E0(self):
        sum = 0.
        M = self.Hcore + self.F
        
        for mu in range(2):
            for nu in range(2):
                sum += self.P[nu][mu] * M[mu][nu]
        
        return 0.5 * sum

    def get_Etot(self):
        pairs = itertools.combinations(self.geometry, 2)
        self.Etot = self.get_E0()

        for pair in pairs:
            dist = np.linalg.norm(pair[0][1] - pair[1][1])
            Z1 = pair[0][0]
            Z2 = pair[1][0]
            self.Etot += Z1 * Z2 / dist
        
        return self.Etot

    def run_scf(self, P_error_tol=1e-4):
        """
        Runs the SCF routine to estimate the ground state of the molecule.
        """
        # Guess P. Default is the zero matrix.

        # Compute S matrix.
        self.get_S()
        
        # Diagonalize S matrix.
        self.diagonalize_S()

        # Compute X matrix.
        self.get_X_symmetric_orthog()

        # Compute Hcore matrix.
        self.get_Hcore()
        
        # Compute G matrix.
        self.get_G()

        # Compute F matrix.
        self.get_F()
       
        """
        print('\nS matrix:')
        print(self.S)
        print('\nT matrix:')
        print(self.T)
        print('\nVnuc matrix:')
        print(self.Vnuc)
        print('\nHcore matrix:')
        print(self.Hcore)
        print('\nG matrix:')
        print(self.G)
        print('\nF matrix:')
        print(self.F)
        print()
        """

        iterations = 0
        print()
        
        # START LOOP.
        while True:
            # Compute orthogonalized F matrix.
            self.get_F_orthog()
            
            # Compute C and e.
            e_new, C_new = self.get_C_and_e()
            
            # Compute P matrix.
            P_new = self.get_P(C_new)
           
            # Evaluate error in density matrices.
            P_error = self.get_P_error(P_new, self.P)
            
            # Update matrices.
            self.orbital_energies = e_new
            self.C = C_new
            self.P = P_new
            self.get_G()
            self.get_F()
            
            iterations += 1
            print('Iteration {}'.format(iterations))

            # Evaluate convergence using density matrix.
            if P_error < P_error_tol:
                print('\nConverged to solution with {} iterations\n'.format(iterations))
                return e_new, C_new


def test_2e_integrals(scf):

    def test_2e_integrals_indv(scf, mu, nu, sigma, lambd):
        sum = 0.

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        coeff = (scf.STO_3G_coeffs[i] 
                                * scf.STO_3G_coeffs[j]
                                * scf.STO_3G_coeffs[k] 
                                * scf.STO_3G_coeffs[l])
                        alpha = scf.STO_3G_exponents[mu][i]
                        beta = scf.STO_3G_exponents[nu][j]
                        gamma = scf.STO_3G_exponents[sigma][k]
                        delta = scf.STO_3G_exponents[lambd][l]
                        RA = scf.geometry[mu][1]
                        RB = scf.geometry[nu][1]
                        RC = scf.geometry[sigma][1]
                        RD = scf.geometry[lambd][1]
                        integral = gaussian_2e(alpha, beta, gamma, delta,
                                                RA, RB, RC, RD)
                        sum += coeff * integral
            
        return sum

    print('\n2e integrals')

    for mu in range(2):
        for nu in range(2):
            for sigma in range(2):
                for lambd in range(2):
                    integral = test_2e_integrals_indv(scf, mu, nu, sigma, lambd)
                    print('{}{}{}{}: {}'.format(mu, nu, sigma, lambd, integral))
    
def test_components(geometry, exponents, n_electrons):
    scf = SCF(geometry, exponents, n_electrons)

    # Print attributes.
    print('\ngeometry:')
    print(scf.geometry)
    
    print('\nSTO-3G exponents:')
    print(scf.STO_3G_exponents)

    print('\nSTO-3G coefficients:')
    print(scf.STO_3G_coeffs)

    # Compute S matrix.
    scf.get_S()
    print('\nS matrix:')
    print(scf.S)

    # Compute T matrix.
    scf.get_T()
    print('\nT matrix:')
    print(scf.T)

    # Compute Vnuc matrices.
    for nucleus in range(2):
        Vnuc_C = scf.get_Vnuc_C(nucleus)
        print('\nVnuc_{} matrix:'.format(nucleus))
        print(Vnuc_C)
    
    scf.get_Vnuc()
    print('\nVnuc matrix:')
    print(scf.Vnuc)

    # Compute Hcore matrix.
    scf.get_Hcore()
    print('\nHcore matrix:')
    print(scf.Hcore)

    # Compute G matrix.
    scf.get_G()
    print('\nG matrix:')
    print(scf.G)

    # Test 2e integrals.
    test_2e_integrals(scf)
    
    # Compute F matrix.
    scf.get_F()
    print('\nF matrix:')
    print(scf.F)

    # Diagonalize S matrix.
    scf.diagonalize_S()
    print('\nDiagonal S matrix:')
    print(scf.S_diag)
    print('\nU matrix:')
    print(scf.U)

    # Compute X matrix.
    scf.get_X_symmetric_orthog()
    print('\nX matrix:')
    print(scf.X)

    # Compute orthogonalized F matrix.
    scf.get_F_orthog()
    print('\nF orthog matrix:')
    print(scf.F_orthog)
    
    # Compute C and e.
    e, C = scf.get_C_and_e()
    print('\nC matrix:')
    print(C)
    print('\ne matrix:')
    print(e)

    # Compute P matrix.
    P = scf.get_P(C)
    print('\nP matrix:')
    print(P)

def test_scf(geometry, exponents, n_electrons):
    scf = SCF(geometry, exponents, n_electrons)
    e, C = scf.run_scf()
    print('\nConverged e matrix:')
    print(e)
    print('\nConverged C matrix:')
    print(C)

    # Get E0.
    E0 = scf.get_E0()
    Etot = scf.get_Etot()
    
    print('\nE0 = {} a.u.'.format(E0))
    print('Etot = {} a.u.'.format(Etot))

if __name__ == '__main__':
    # Distances in a.u.
    # H2
    geometry_H2 = [
                    (1., np.array([0., 0., 0.])),
                    (1., np.array([1.4, 0., 0.]))
                  ]
    
    exponents_H2 = np.array([
                             [0.168856, 0.623913, 3.42525],
                             [0.168856, 0.623913, 3.42525]
                            ])
    
    n_electrons_H2 = 2

    test_components(geometry_H2, exponents_H2, n_electrons_H2)    
    test_scf(geometry_H2, exponents_H2, n_electrons_H2)

    # HeH+
    geometry_HeH = [
                    (1., np.array([0., 0., 0.])),
                    (2., np.array([1.4632, 0., 0.]))
                   ]
    
    zeta_H = 1.24
    zeta_He = 2.0925
    alpha1 = np.array([0.109818, 0.405771, 2.22766])

    exponents_HeH = np.array([alpha1 * zeta_H**2, alpha1 * zeta_He**2])
    
    n_electrons_HeH = 2

    test_components(geometry_HeH, exponents_HeH, n_electrons_HeH)    
    test_scf(geometry_HeH, exponents_HeH, n_electrons_HeH)


