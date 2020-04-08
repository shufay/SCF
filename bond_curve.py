import scf
import matplotlib.pyplot as plt
import numpy as np

def iterate_bond_lengths(Z1, Z2, exponents, min, max, n):
    bond_lengths = np.linspace(min, max, n)
    n_electrons = 2
    
    Etots = np.zeros(len(bond_lengths))
    
    for i, bond_length in enumerate(bond_lengths):
        geometry = [
                    (Z1, np.array([0., 0., 0.])),
                    (Z2, np.array([bond_length, 0., 0.]))
                   ]

        scf_ = scf.SCF(geometry, exponents, n_electrons)
        scf_.run_scf()
        Etots[i] = scf_.get_Etot()
        
    return bond_lengths, Etots

# Run.
EH2 = - 2. * 0.4666
exponents_H2 = np.array([
                      [0.168856, 0.623913, 3.42525],
                      [0.168856, 0.623913, 3.42525]
                     ])

bond_lengths_H2, Etots_H2 = iterate_bond_lengths(1, 1, exponents_H2, 
                                                 0.5, 6.0, 50)
plt.plot(bond_lengths_H2, Etots_H2 - EH2)
plt.xlabel('R (a.u.)')
plt.ylabel('E(H2) - 2E(H) (a.u.)')
plt.title('SCF H2')
plt.tight_layout()
plt.savefig('H2.png')
plt.show()


EHe = -2.643876
zeta_H = 1.24
zeta_He = 2.0925
alpha1 = np.array([0.109818, 0.405771, 2.22766])
exponents_HeH = np.array([alpha1 * zeta_H**2, alpha1 * zeta_He**2])

bond_lengths_HeH, Etots_HeH = iterate_bond_lengths(1, 2, exponents_HeH, 
                                                   0.6, 3.5, 40)
plt.plot(bond_lengths_HeH, Etots_HeH - EHe)
plt.xlabel('R (a.u.)')
plt.ylabel('E(HeH+) - E(He) (a.u.)')
plt.title('SCF HeH+')
plt.tight_layout()
plt.savefig('HeH+.png')
plt.show()
