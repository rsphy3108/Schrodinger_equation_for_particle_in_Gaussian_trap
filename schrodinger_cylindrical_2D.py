import numpy as np
from scipy import sparse, linalg, integrate
from scipy.sparse import linalg as sla
from scipy import sparse
import scipy.fft as spfft
    
def schrodinger2D(rmax, Nr, zmin, zmax, Nz, Vfun2D, params, neigs, hbar = 1, m = 1, E0 = 0.0, l = 0,findpsi = False):
    """
    Solves the radial and the axial part of a Cylindrically Symmetric Schrodinger equation numerically using the finite element method.
    Inputs
    ------

    rmax: float
        maximum value of the r axis
    Nr: int
        number of finite elements in the r axis
    zmin: float
        minimum value of the z axis
    zmax: float
        maximum value of the z axis
    Nz: int
        number of finite elements in the z axis        
    Vfun2D: function
        potential energy function
    params: list
        list containing the parameters of Vfun
    neigs: int
        number of eigenvalues to find
    E0: float
        eigenenergy value to solve for
    hbar: float
        Plank's constant
    m: float
        mass of the particle
    l: Angular momentum
    findpsi: bool
        If True, the eigen wavefunctions will be calculated and returned.
        If False, only the eigen energies will be found.
    
    
    Returns
    -------
    evl: np.array
        eigenvalues
    evt: np.array
        eigenvectors
    r: np.array
        r axis values
    z: np.array
        z axis values
    """
    r = np.linspace(0.00001*rmax, rmax, Nr)  
    dr = r[1] - r[0]  
    z = np.linspace(zmin, zmax, Nz)
    dz = z[1] - z[0]

    V = Vfun2D(r, z, params)

    # create the 2D Hamiltonian matrix
    
    #Hamiltonian in radial direction
    #Kinetic energy part, away from the origin and for dr small we can neglect the first order derivative term and only use the second order part.
    # See also https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf 
    Hr1 = create_hamiltonian1(Nr, dr,NBC = True)
    #Hamiltonian due to angular momentum
    Hr2 = sparse.csr_matrix(np.diag(-l**2/r**2))
    Hr = Hr1 + Hr2
    
    #Hamiltonian in z direction as usual
    #Kinetic energy part
    Hz = create_hamiltonian1(Nz, dz)

    Ir = sparse.eye(Nr, Nr)
    Iz = sparse.eye(Nz, Nz)
    H = sparse.kron(Hr, Iz) + sparse.kron(Ir, Hz)  
    H = -(0.5*(hbar**2)/m)*H
    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nr * Nz):
        H[i, i] = H[i, i] + V[i]    

    # convert to csc form and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k = neigs, sigma = E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, r, z

def create_hamiltonian1(Nx, dx,NBC = False):
    """
    Creates a 1 dimensional Hamiltonian.
    see https://en.wikipedia.org/wiki/Finite_difference
    Inputs
    ------
    Nx: int
        number of elements in that axis
    dx: float
        step size
       
    Returns
    -------
    H: np.array
        np.array of the Hamiltonian
    """

    H = sparse.diags([1, -2, 1], 
                      [-1, 0, 1],
                        shape=(Nx, Nx)) / dx**2

    if NBC:
        H = H.toarray()
        H[0,1] = 2/(dx**2)
        H = sparse.csr_matrix(H)
    return H

def schrodinger2D_TD_kspace(karraysq,psi0,timevars, hbar = 1, m = 1):  
    """
    Solves the time dependent free particle hamiltonian in the fourier space 
    
    Inputs
    ------
    psi0: 2D array
          the initial wavefunction in the position space
    karraysq : 2D array
          The array of wavevector squared
    timevars: array
            time parameters. timevars[0] - intial tim, timevars[1]- final time, timevars[3] - time                   interval.
    hbar: float
        Plank's constant
    m: float
        mass of the particle
       
    Returns
    -------
    t_eval: np.array
        times at which wavevector is stored. 
    psit_all:
    the stored wavevectors for the times in t_save  
    """
    t0 = timevars[0]    # initial time
    tf = timevars[1]    # final time
    dt = timevars[2]  # time interval for evaluation
    Nt = int((tf-t0)/dt)
    t_eval = np.arange(t0, tf, dt)  # recorded time shots
    parraysqarray = ((hbar**2)*karraysq)*(1/(2*m))*(-1j/hbar)*dt
    psit_all = []
    for k in range(Nt):
        psit = np.multiply(np.exp(parraysqarray*k),psi0)
        psitifft =  spfft.fftn(psit,axes = [0,1,2])
        psitifft = psitifft/np.sqrt(np.sum(np.abs(psitifft)**2))
        psit_all.append(psitifft.copy())
            
    return t_eval, np.array(psit_all)