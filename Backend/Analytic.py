import numpy as np
import Backend.Constants as cst



#=========================================
# DIPOLAR EFFECT
#=========================================
def get_x_co(mux_s,betx_s,I,xw,yw,Qx,mu0 = 0,E = 450e9,L=1):
    # Confirming that mu and bet were given as vectors
    assert(np.shape(mux_s)==np.shape(betx_s))
    
    # Constants calculations
    Brho   = np.sqrt(E**2-cst.m_p_eV**2)/cst.c
    kick_0 = cst.mu0*(I*L)/(Brho*2*np.pi)
    kick   = kick_0*(2*xw/(xw**2+yw**2))
    
    # Finding beta at location of kick
    bet0 = betx_s[np.argmin(np.abs(mux_s-mu0))] 
    Amp  = np.sqrt(betx_s*bet0/(4*np.sin(np.pi*Qx)**2)) * kick
    
    return Amp*np.cos(2*np.pi*(np.abs(mux_s-mu0)) - np.pi*Qx)


def get_y_co(muy_s,bety_s,I,xw,yw,Qy,mu0 = 0,E = 450e9,L=1):
    # Confirming that mu and bet were given as vectors
    assert(np.shape(muy_s)==np.shape(bety_s))
    
    # Constants calculations
    Brho   = np.sqrt(E**2-cst.m_p_eV**2)/cst.c
    kick_0 = cst.mu0*(I*L)/(Brho*2*np.pi)
    kick   = kick_0*(2*yw/(xw**2+yw**2))
    
    # Finding beta at location of kick
    bet0 = bety_s[np.argmin(np.abs(muy_s-mu0))] 
    Amp  = np.sqrt(bety_s*bet0/(4*np.sin(np.pi*Qy)**2)) * kick
    
    return Amp*np.cos(2*np.pi*(np.abs(muy_s-mu0)) - np.pi*Qy)
#==========================================



#=========================================
# OCTUPOLAR EFFECT
#=========================================
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def dQxdQy(I, xw , yw , betx, bety, Jx = 0 , Jy = 0 ,E = 450e9,L=1):
    rw,phiw = cart2pol(xw, yw)
    
    Brho = np.sqrt(E**2-cst.m_p_eV**2)/cst.c
    Q0   = -cst.mu0*(I*L)/(Brho*8*np.pi**2)
    
    quad = 2*np.cos(2*phiw)/rw**2
    octu = 2*np.cos(4*phiw)/rw**4 * (3/2)
    
    dQx  = Q0*( 1*quad*betx + octu*(betx**2*Jx - 2*betx*bety*Jy))
    dQy  = Q0*(-1*quad*bety + octu*(bety**2*Jy - 2*bety*betx*Jx))
    
    return dQx,dQy

def dQxfit(I,A,betx,E = 450e9,L=1):
    Brho = np.sqrt(E**2-cst.m_p_eV**2)/cst.c
    Q0   = -cst.mu0*L*betx/(Brho*8*np.pi**2)
    return Q0*A*I

def dQyfit(I,A,bety,E = 450e9,L=1):
    Brho = np.sqrt(E**2-cst.m_p_eV**2)/cst.c
    Q0   = -cst.mu0*L*bety/(Brho*8*np.pi**2)
    return -Q0*A*I

def extractOffset(A,xw=None,yw=None):
    
    if yw is not None:
        pref = np.sqrt(1-4*A*yw**2)/A
        return np.sqrt(-pref + 1/A - yw**2)#,np.sqrt(pref + 1/A - yw**2))
    
    if xw is not None:
        pref = np.sqrt(4*A*xw**2+1)/A
        return np.sqrt(pref - 1/A - xw**2)#(np.sqrt(-pref - 1/A - xw**2),)
    
#==========================================