import numpy as np
from scipy import sparse, linalg, integrate
from scipy.sparse import linalg as sla
from scipy import sparse
import scipy.fft as spfft
from scipy import interpolate
    
def schrodinger2D(x,y, Vfun2D, params, neigs, hbar = 1, m = 1, E0 = 0.0,findpsi = False):
    """
    Diagonalizes the 2 dimensional Schrodinger equation numerically to find eigenfunctions using the finite element method.
    Inputs
    ------

    x: np.array
        vector of grid in x-direction 
    y: np.array
        vector of grid in y-direction     
    Vfun2D: function
        potential energy function
    params: list
        list containing the parameters of Vfun2D
    neigs: int
        number of eigenvalues to find
    E0: float
        eigenenergy value to solve for
    findpsi: bool
        If True, the eigen wavefunctions will be calculated and returned.
        If False, only the eigen energies will be found.
    hbar: float
        Plank's constant
    m: float
        mass of the particle

    Returns
    -------
    evl: np.array
        eigenvalues
    evt: np.array
        eigenvectors
    """
    dx = x[1] - x[0]  
    dy = y[1] - y[0] 

    V = Vfun2D(x, y, params)

    # create the 2D Hamiltonian matrix
    Nx = len(x)
    Ny = len(y)
    Hx = create_hamiltonian1(Nx, dx,NBC = False)
    Hy = create_hamiltonian1(Ny, dy,NBC = False)

    Ix = sparse.eye(Nx, Nx)
    Iy = sparse.eye(Ny, Ny)
    H = sparse.kron(Hx, Iy) + sparse.kron(Ix, Hy)  
    H = -(0.5*(hbar**2)/m)*H
    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]    

    # convert to csc form and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k = neigs, sigma = E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt

def create_hamiltonian1(Nx, dx,NBC = False):
    """
    Creates a 1 dimensional Hamiltonian for the kinetic energy operator.
    see https://en.wikipedia.org/wiki/Finite_difference
    Inputs
    ------
    Nx: int
        number of elements in that axis
    dx: float
        step size
    NBC:bool
        whether to use the Neuman Boundary condition or not.
       
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

def TD_Unitary(x,y,time,Vfun2D, params, hbar = 1, m = 1):
    """
    Provides the time dependent unitary operator for solving the time-dependent Schrodinger equantion.
    ------

    x: np.array
        vector of grid in x-direction 
    y: np.array
        vector of grid in y-direction     
    Vfun2D: function
        potential energy function
    params: list
        list containing the parameters of Vfun2D
    hbar: float
        Plank's constant
    m: float
        mass of the particle
    Returns
    -------
    U: np.array
        Unitary matrix
    """
    Nx = len(x)
    Ny = len(y)

    # RHS of Schrodinger Equation
    dx = x[1] - x[0]  
    dy = y[1] - y[0] 

    V = Vfun2D(x, y, params)

    # create the 2D Hamiltonian matrix

    Hx = create_hamiltonian1(Nx, dx,NBC = False)
    Hy = create_hamiltonian1(Ny, dy,NBC = False)

    Ix = sparse.eye(Nx, Nx)
    Iy = sparse.eye(Ny, Ny)
    It = sparse.kron(Ix, Iy)
    H = sparse.kron(Hx, Iy) + sparse.kron(Ix, Hy)  
    H = -(0.5*(hbar**2)/m)*H
    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]
    
     
    H = -(1j/hbar)*H*time 
    H = H.tocsc()
    U = sparse.linalg.expm(H)
    U = U.tocsc()
    return U

def schrodinger2D_TD(x,y,psi0,timevars,Vfun2D, params, hbar = 1, m = 1):
    """
    Crank-Nicolson method for solving the time dependent Schrodinger equantion.
    See https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method
    https://physics.stackexchange.com/questions/12199/solving-schr%C3%B6dingers-equation-with-crank-nicolson-method
    ------

    x: np.array
        vector of grid in x-direction 
    y: np.array
        vector of grid in y-direction
    psi0: np.array
	intital wavefunction 	     
    timevars: list
	list specifying the times at which to solve the equation	
    Vfun2D: function
        potential energy function
    params: list
        list containing the parameters of Vfun2D
    hbar: float
        Plank's constant
    m: float
        mass of the particle
    Returns
    -------
    t_save: np.array
        times at which wavevector is stored. 
    psit_all:
	the stored wavevectors for the times in t_save  
    """

    Nx = len(x)
    Ny = len(y)

    dt_save = timevars[3]  # time interval for snapshots
    t0 = timevars[0]    # initial time
    tf = timevars[1]    # final time
    dt = timevars[2]  # time interval for evaluation of schrodinger equation

    Nt = int((tf-t0)/dt)
    Nt_save = int((tf-t0)/dt_save)
    t_eval = np.arange(t0, tf, dt)  # recorded time shots
    t_save = np.arange(t0, tf, dt_save)  # recorded time shots

    dx = x[1] - x[0]  
    dy = y[1] - y[0] 

    V = Vfun2D(x, y, params)

    # create the 2D Hamiltonian matrix
    Hx = create_hamiltonian1(Nx, dx,NBC = False)
    Hy = create_hamiltonian1(Ny, dy,NBC = False)

    Ix = sparse.eye(Nx, Nx)
    Iy = sparse.eye(Ny, Ny)
    It = sparse.kron(Ix, Iy)
    H = sparse.kron(Hx, Iy) + sparse.kron(Ix, Hy)  
    H = -(0.5*(hbar**2)/m)*H
    # Convert to lil form and add potential energy function
    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]

    H = -1j*dt/ hbar *H
	
    #Use Crank-Nikolson method
	
    Hminus = It - H
    Hplus =  It + H
    # Convert to lil form and add potential energy function
    Hminus = Hminus.tolil()
    Hplus = Hplus.tolil()
    
    Hminus = Hminus.tocsc() 
    Hplus = Hplus.tocsc() 

    psit_all = []
     
    #Advance the wavevector 	
    for k in range(Nt):
        psit = sparse.linalg.spsolve(Hminus, Hplus.dot(psi0))
        psi0 = psit.copy()
        if k%int((Nt/Nt_save))==0:
            psit_all.append(psit)
            
           
    return t_save, np.array(psit_all)

def Splitstep_2D_TD(x,y,psi0,timevars,Vfun2D, params,karraysq, hbar = 1, m = 1):
    """
    Split-Step fourier transform method for solving the time dependent Schrodinger equantion.
    see https://en.wikipedia.org/wiki/Split-step_method
    ------

    x: np.array
        vector of grid in x-direction 
    y: np.array
        vector of grid in y-direction
    psi0: np.array
	intital wavefunction 	     
    timevars: list
	list specifying the times at which to solve the equation	
    Vfun2D: function
        potential energy function
    params: list
        list containing the parameters of Vfun2D
    karraysq:np.array
	2D array of values of the momentum values, used for multiplying the fourier transformed wavevector
    hbar: float
        Plank's constant
    m: float
        mass of the particle
    Returns
    -------
    t_save: np.array
        times at which wavevector is stored. 
    psit_all:
	the stored wavevectors for the times in t_save  
    """
    Nx = len(x)
    Ny = len(y)

    dt_save = timevars[3]  # time interval for snapshots
    t0 = timevars[0]    # initial time
    tf = timevars[1]    # final time
    dt = timevars[2]  # time interval for evaluation

    Nt = int((tf-t0)/dt)
    Nt_save = int((tf-t0)/dt_save)
    t_eval = np.arange(t0, tf, dt)  # recorded time shots
    t_save = np.arange(t0, tf, dt_save)  # recorded time shots


    dx = x[1] - x[0]  
    dy = y[1] - y[0] 
    

    V = Vfun2D(x, y, params)*(-1j/hbar)*dt
    Vexp = np.exp(V)  
  
    parraysqarray = ((hbar**2)*karraysq)*(1/(2*m))*(-1j/hbar)*dt
    kexp = np.exp(parraysqarray)

    psit_all = []
    #Advance the wavevector 
    for k in range(Nt):
    	psix =  np.multiply(Vexp,psi0)
    	psik = np.multiply(kexp,spfft.fftn(psix,axes = [0,1]))
    	psit = spfft.ifftn(psik,axes = [0,1])
    	psit = psit/np.sqrt(np.sum(np.abs(psit)**2))
    	psi0 = psit.copy()
    	if k%int((Nt/Nt_save))==0:
        	psit_all.append(psit)
            
           
    return t_save, np.array(psit_all)

def get_k_array(xarray,yarray,samplerate):
    """
    This function gives the squared wavevector array for the fourier transforms
    Inputs
    ------  
    xarray: 1d array
        cordinates in x-direction
    yarray: 1d array
        cordinates in y-direction
    
    Returns
    -------
    karaysq: 2D array
            Square of the wavenumber values 
    """
    
    freqx = spfft.fftfreq(xarray.shape[0], d=1/samplerate)*(2*np.pi)
    freqx = np.sort(freqx/(-2*xarray[0]))
    freqy = spfft.fftfreq(yarray.shape[0], d=1/samplerate)*(2*np.pi)
    freqy = np.sort(freqy/(-2*yarray[0]))
    
    
    karaysq = []
    for fx in spfft.ifftshift(freqx):
        arraytemp1 = []
        for fy in spfft.ifftshift(freqy):
            arraytemp1.append(fx**2+fy**2)
        karaysq.append(arraytemp1)
    karaysq = np.array(karaysq)
    
    return karaysq


def get_extrapolated_wfn(psi,xlist,ylist,xs,ys):
    """
    Takes a wavefunction specified at points xlist and ylist, interpolates it and specifies the wavefunction at points xs,ys
    ------  
   
    psi: 2D array
        wavefunction at xlist and ylist
    
    Returns
    -------
    psinew: 2D array
        wavefunction at xs and ys
    """
        
    fnintp = interpolate.RectBivariateSpline(xlist,ylist, np.real(psi))
    psinew = fnintp(xs,ys)
    psinew = psinew/np.sqrt(np.sum(np.abs(psinew)**2))
    return psinew

def schrodinger2D_kspace_TD(psifft,karraysq,timevars, hbar = 1, m = 1):
    """
    Solves the time dependent free particle hamiltonian in the fourier space   
    Inputs
    ------
    psi0: 2D array
          the initial wavefunction in the position space
    karraysq : 2D array
          The array of wavevector squared
    timevars: array
            time parameters. timevars[0] - intial tim, timevars[1]- final time, timevars[3] - time interval.
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
    parraysqarray = ((hbar**2)*karraysq)*(1/(2*m))*(-1j/hbar)*dt #d(psi)/dt = 
    #psit = psi0.copy()
    
    psit_all = []
    for k in range(Nt):
        psit = np.multiply(np.exp(parraysqarray*k),psifft)
        #psi0 = psit.copy()
        psitifft =  spfft.ifftn(psit,axes = [0,1])
        psitifft = psitifft/np.sqrt(np.sum(np.abs(psitifft)**2))
        psit_all.append(psitifft.copy())
            
    return t_eval, np.array(psit_all)