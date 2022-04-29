# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Common functions used in the variational formulations for the variational forms
of the mesh lifting equations, fluid equations and the structure equation.
"""

from dolfin import grad, det, Identity, tr, inv, inner, as_tensor, as_vector, derivative, variable
import ufl  # ufl module

def get_dimesion(u):

    # Get dimension of tensor
    try:
        from ufl.domain import find_geometric_dimension

        dim = find_geometric_dimension(u)

    except:

        try:
            dim = len(u)
        except:
            dim = 3

    return dim

def F_(d):
    """
    Deformation gradient tensor
    """
    return Identity(get_dimesion(d)) + grad(d)


def J_(d):
    """
    Determinant of the deformation gradient
    """
    return det(F_(d))


def eps(d):
    """
    Infinitesimal strain tensor
    """
    return 0.5 * (grad(d) * inv(F_(d)) + inv(F_(d)).T * grad(d).T)


def sigma_f_u(u, d, mu_f):
    """
    Deviatoric component of the Cauchy stress tensor (fluid problem)
    """
    return mu_f * (grad(u) * inv(F_(d)) + inv(F_(d)).T * grad(u).T)


def sigma_f_p(p, u):
    """
    Hydrostatic component of the Cauchy stress tensor (fluid problem)
    """
    return -p * Identity(len(u))


def sigma(u, p, d, mu_f):
    """
    Cauchy stress tensor (fluid problem)
    """
    return sigma_f_u(u, d, mu_f) + sigma_f_p(p, u)


def E(d):
    """
    Green-Lagrange strain tensor
    """
    return 0.5*(F_(d).T*F_(d) - Identity(len(d)))


def S(d, material_parameters):
    """
    Second Piola-Kirchhoff Stress (solid problem)
    """
    F = F_(d)
    S__ = inv(F)*Piola1(d, material_parameters)

    return S__

def Piola1(d, material_parameters):
    """
    First Piola-Kirchhoff Stress (solid problem)
    """
    if material_parameters["material_model"] == "StVenantKirchoff":
        I = Identity(len(d)) # Identity matrix
        lambda_s = material_parameters["lambda_s"]
        mu_s = material_parameters["mu_s"]
        S_svk = 2*mu_s*E(d) + lambda_s*tr(E(d))*I  # Calculate First Piola Kirchoff Stress with Explicit form of St. Venant Kirchoff model
        P = F_(d)*S_svk  # Convert to First Piola-Kirchoff Stress
    else: 
        # ["StVenantKirchoff",""StVenantKirchoffEnergy","NeoHookean","MooneyRivlin","Gent"]
        F = ufl.variable(F_(d))  # Note that dolfin and ufl "variable" are different.
        if material_parameters["material_model"] == "StVenantKirchoffEnergy":
            W = W_St_Venant_Kirchoff(F, material_parameters["lambda_s"], material_parameters["mu_s"])
        elif material_parameters["material_model"] == "NeoHookean":
            W = W_Neo_Hookean(F, material_parameters["lambda_s"], material_parameters["mu_s"])  
        elif material_parameters["material_model"] == "MooneyRivlin":
            W = W_Mooney_Rivlin(F, material_parameters["lambda_s"], material_parameters["mu_s"], material_parameters["C01"], material_parameters["C10"], material_parameters["C11"]) 
        elif material_parameters["material_model"] == "Gent":
            W = W_Gent(F, material_parameters["mu_s"], material_parameters["Jm"])  
        elif material_parameters["material_model"] == "Exponential":
            W = W_Exponential(F, material_parameters["C01"], material_parameters["C02"])  
        else:
            print('Invalid entry for material_model, choose from ["StVenantKirchoff",""StVenantKirchoffEnergy","NeoHookean","MooneyRivlin","Gent","Exponential"]')
        
        P = ufl.diff(W, F) # First Piola-Kirchoff Stress for compressible hyperelastic material (https://en.wikipedia.org/wiki/Hyperelastic_material)
    
    return P


def S_linear(d, alfa_mu, alfa_lam):
    """
    Second Piola-Kirchhoff Stress (mesh problem - Linear Elastic materials)
    """
    return alfa_lam * tr(eps(d)) * Identity(len(d)) + 2.0 * alfa_mu * eps(d)



def get_eig(T):
########################################################################
# Method for the analytical calculation of eigenvalues for 3D-Problems #
# from: https://fenicsproject.discourse.group/t/hyperelastic-model-problems-on-plotting-stresses/3130/6
########################################################################
    '''
    Analytically calculate eigenvalues for a three-dimensional tensor T with a
    characteristic polynomial equation of the form

                lambda**3 - I1*lambda**2 + I2*lambda - I3 = 0   .

    Since the characteristic polynomial is in its normal form , the eigenvalues
    can be determined using Cardanos formula. This algorithm is based on:
    "Efficient numerical diagonalization of hermitian 3 × 3 matrices" by
    J. Kopp (equations: 21-34, with coefficients: c2=-I1, c1=I2, c0=-I3).

    NOTE:
    The method implemented here, implicitly assumes that the polynomial has
    only real roots, since imaginary ones should not occur in this use case.

    In order to ensure eigenvalues with algebraic multiplicity of 1, the idea
    of numerical perturbations is adopted from "Computation of isotropic tensor
    functions" by C. Miehe (1993). Since direct comparisons with conditionals
    have proven to be very slow, not the eigenvalues but the coefficients
    occuring during the calculation of them are perturbated to get distinct
    values.
    '''

    # determine perturbation from tolerance
    tol = 1e-8
    pert = 2*tol

    # get required invariants
    I1 = tr(T)                                                               # trace of tensor
    I2 = 0.5*(tr(T)**2-inner(T,T))                                        # 2nd invariant of tensor
    I3 = det(T)                                                              # determinant of tensor

    # determine terms p and q according to the paper
    # -> Follow the argumentation within the paper, to see why p must be
    # -> positive. Additionally ensure non-zero denominators to avoid problems
    # -> during the automatic differentiation
    p = I1**2 - 3*I2                                                            # preliminary value for p
    p = ufl.conditional(ufl.lt(p,tol),abs(p)+pert,p)                            # add numerical perturbation to p, if close to zero; ensure positiveness of p
    q = 27/2*I3 + I1**3 - 9/2*I1*I2                                             # preliminary value for q
    q = ufl.conditional(ufl.lt(abs(q),tol),q+ufl.sign(q)*pert,q)                # add numerical perturbation (with sign) to value of q, if close to zero

    # determine angle phi for calculation of roots
    phiNom2 =  27*( 1/4*I2**2*(p-I2) + I3*(27/4*I3-q) )                         # preliminary value for squared nominator of expression for angle phi
    phiNom2 = ufl.conditional(ufl.lt(phiNom2,tol),abs(phiNom2)+pert,phiNom2)    # add numerical perturbation to ensure non-zero nominator expression for angle phi
    phi = 1/3*ufl.atan_2(ufl.sqrt(phiNom2),q)                                   # calculate angle phi

    # calculate polynomial roots
    lambda1 = 1/3*(ufl.sqrt(p)*2*ufl.cos(phi)+I1)
    lambda2 = 1/3*(-ufl.sqrt(p)*(ufl.cos(phi)+ufl.sqrt(3)*ufl.sin(phi))+I1)
    lambda3 = 1/3*(-ufl.sqrt(p)*(ufl.cos(phi)-ufl.sqrt(3)*ufl.sin(phi))+I1)
    
    # Eigenvector subrouting (Doesn't work yet, needs special cases implemented)
    #eigvec1 = eigvec(lambda1,T,tol)
    #eigvec2 = eigvec(lambda2,T,tol)
    #eigvec3 = eigvec(lambda3,T,tol)


    # return polynomial roots (eigenvalues)
    #eig = as_tensor([[lambda1 ,0 ,0],[0 ,lambda2 ,0],[0 ,0 ,lambda3]])

    return lambda1, lambda2, lambda3 #, eigvec1, eigvec2, eigvec3


# We could also add the Yeoh model, or Fung model if its possible to make it compressible.


def W_St_Venant_Kirchoff(F, lambda_s, mu_s):
    """
    Strain energy density, St. Venant Kirchoff Material
    """

    E_ = 0.5*(F.T*F - Identity(get_dimesion(F)))
    J = det(F)

    W = lambda_s / 2 * (tr(E_) ** 2) + mu_s * tr(E_*E_)

    return W


def W_Neo_Hookean(F, lambda_s, mu_s):
    """
    Strain energy density, Neo-Hookean Material
    """
    C1 = mu_s/2
    D1 = lambda_s/2
    C = F.T * F  # Right cauchy-green strain tensor
    I1 = tr(C)
    J = det(F)

    W = C1*(I1 - get_dimesion(F) - 2*ufl.ln(J)) + D1*(J-1)**2

    return W

def W_Gent(F, mu_s, Jm):  # Test with skin material properties
    """
    Strain energy density, Compressible Gent Material
    As described in https://www.researchgate.net/profile/Aflah-Elouneg/publication/353259552_An_open-source_FEniCS-based_framework_for_hyperelastic_parameter_estimation_from_noisy_full-field_data_Application_to_heterogeneous_soft_tissues/links/6124e7c71e95fe241af14697/An-open-source-FEniCS-based-framework-for-hyperelastic-parameter-estimation-from-noisy-full-field-data-Application-to-heterogeneous-soft-tissues.pdf?origin=publication_detail
    "An open-source FEniCS-based framework for hyperelastic parameter estimation from noisy full-field data: Application to heterogeneous soft tissues"
    """

    B = F*F.T  # Left cauchy-green strain tensor
    I1 = tr(B)
    J = det(F)

    W = -(mu_s/2)*( Jm * ufl.ln( 1 - (I1-3)/Jm ) + 2*ufl.ln(J) )

    return W

def W_Mooney_Rivlin(F, lambda_s, mu_s, C01, C10, C11):
    """
    Strain energy density, Compressible Mooney-Rivlin Material
    following: https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
    """
    K = lambda_s + 2*mu_s/3          # Compute bulk modulus from lambda and mu
    D1 = 2/K                         # D1 is calculated from the Bulk Modulus
    B = F*F.T                        # Left cauchy-green strain tensor
    I1 = tr(B)                       # 1st Invariant
    I2 = 0.5*(tr(B)**2-tr(B * B))    # 2nd invariant (also known as I3)
    Ibar1 = (J**(-2/3))*I1
    Ibar2 = (J**(-4/3))*I2
    # Strain energy density function for 3 term Compressible Mooney-Rivlin Model
    W = C01*(Ibar2-3) + C10*(Ibar1-3) + C11*(Ibar2-3)*(Ibar1-3) + (1/D1)*(J-1)**2   
    
    return W

def W_Exponential(F, C01, C02):  
    """
    Strain energy density, Exponential
    As described in "Diversity in the Strength and Structure of Unruptured Cerebral Aneurysms"
    This model is incompressible and doesn't run yet - investigate compressible versions
    """

    B = F*F.T  # Left cauchy-green strain tensor
    I1 = tr(B)
    J = det(F)
    # Actual equation                                   ) + (compressibility)
    K = 2000 # Pa
    W = (C01/(2*C02)) * ( ufl.exp( C02 * (I1-3)**2 ) - 1) + (K)*(J-1)**2
    #
    #  NEED to look into compressibility - good reference "Hyperelastic Energy Densities for Soft Biological Tissues: A Review"
    #

    return W

'''
These functions do not work yet, but close to working so I don't want to comletely abandon them. The Ogden model works with applied deformation gradients. 
'''


def eigvec(Lam,A,tol):
    '''
    Analytically calculate eigenvectors for a three-dimensional tensor T with a
    characteristic polynomial equation of the form

                lambda**3 - I1*lambda**2 + I2*lambda - I3 = 0   .
    This algorithm is based on:
    "Efficient numerical diagonalization of hermitian 3 × 3 matrices" by
    J. Kopp (equations: 36-40).

    '''

    # This may be the slow part of the code. Let's see if it's possible to reduce the number of conditionals here. 
    # Maybe we should lower the tolerance (1e-12 or similar)

    I = as_tensor([[1,0,0],[0,1,0],[0,0,1]])
    A_m_LamI1 = A[:,0]-Lam*I[:,0]
    A_m_LamI2 = A[:,1]-Lam*I[:,1]

    mu = ufl.conditional(ufl.lt(abs(A_m_LamI2[0]),tol), 100000000 , A_m_LamI1[0]/A_m_LamI2[0]) # Check use of gt or lt... not sure. gt seems closer but doesn't make any sense.
    special_case = ufl.conditional(ufl.lt(abs(A_m_LamI2[0]),tol), 100000000 , ufl.conditional(ufl.lt(abs(A_m_LamI2[1]),tol), 100000000 , abs(A_m_LamI1[0]/A_m_LamI2[0] - A_m_LamI1[1]/A_m_LamI2[1])))
    v = ufl.conditional(ufl.lt(special_case,tol),(1/ufl.sqrt(1+mu**2))*as_vector([1,-mu,0]),ufl.operators.cross(A_m_LamI1,A_m_LamI2)) # decide between formula (40) or formula (39)

    #try:
    #    mu = A_m_LamI1[0]/A_m_LamI2[0] 
    #    special_case = abs(A_m_LamI1[0]/A_m_LamI2[0] - A_m_LamI1[1]/A_m_LamI2[1])
    #    #v = ufl.conditional(ufl.lt(special_case,tol),(1/ufl.sqrt(1+mu**2))*as_vector([1,-mu,0]),ufl.operators.cross(A_m_LamI1,A_m_LamI2)) # Check use of gt or lt... not sure. gt seems closer but doesn't make any sense. 

    #    v = ufl.conditional(ufl.lt(special_case,tol),(1/ufl.sqrt(1+mu**2))*as_vector([1,-mu,0]),ufl.operators.cross(A_m_LamI1,A_m_LamI2)) # decide between formula (40) or formula (39)
    #except:
    #    v = ufl.operators.cross(A_m_LamI1,A_m_LamI2) # use formula (39) if there is a divide by zero
    magnitude = ufl.sqrt(v[0]**2+v[1]**2+v[2]**2)
    v = ufl.conditional(ufl.lt(magnitude,tol),as_vector([1,1,1]),v/magnitude) # if eigenvector is zero, input arbitrary vector because it will be multiplied by zero anyway. Otherwise normalize by the magnitude
    # This arbitraty vector may be the sketchiest part of this code. May be better to input an identy matrix of eigenvalues?
    return v


def S_ogden(d,  muO1=0.64724, alO1=1.3, muO2=0.001177,alO2=5.0, K=200000):
    """
    Second Piola-Kirchhoff Stress (solid problem - 2-term Ogden material)
    This subroutine doesn't work yet, due to the lack of a reliable eigenvector subroutine to rotate the stress tensor. 
    If a reliable eigenvector subroutine was developed, the ogden model would work. 
    """
    I = Identity(len(d))
    F = F_(d)
    #F = as_tensor([[1,0.2,0],[0,1,0],[0,0,1]]).T # As tensor seems to work the opposite way with vectors compared to numpy arrays... need to transpose
    C = F.T * F  # Right cauchy-green strain tensor
    eigC1, eigC2, eigC3, vC1, vC2, vC3 = get_eig(C) # Eigenvalues of C
    eigT = as_tensor([[eigC1,0,0],[0,eigC2,0],[0,0,eigC3]])
    #eigVT = as_tensor([vC1, vC2, vC3]).T

    # Eigenvalues of U (The principal stretches) are the square root of the eigenvalues of C
    lambda1 = ufl.sqrt(eigC1)
    lambda2 = ufl.sqrt(eigC2)
    lambda3 = ufl.sqrt(eigC3)

    J = lambda1*lambda2*lambda3 # Incompressibility parameter
    lambda1_s=lambda1/J**(1/3) # Calculate volumetric-independant principal stretches
    lambda2_s=lambda2/J**(1/3) # Calculate volumetric-independant principal stretches
    lambda3_s=lambda3/J**(1/3) # Calculate volumetric-independant principal stretches

    # Calculate am
    amO1=lambda1_s**alO1+lambda2_s**alO1+lambda3_s**alO1
    amO2=lambda1_s**alO2+lambda2_s**alO2+lambda3_s**alO2

    # Calculate Principal Kirchoff stresses
    TauP1= muO1*(lambda1_s**alO1 - (1/3)*amO1) + muO2*(lambda1_s**alO2 - (1/3)*amO2)+K*J*(J-1)  
    TauP2= muO1*(lambda2_s**alO1 - (1/3)*amO1) + muO2*(lambda2_s**alO2 - (1/3)*amO2)+K*J*(J-1) 
    TauP3= muO1*(lambda3_s**alO1 - (1/3)*amO1) + muO2*(lambda3_s**alO2 - (1/3)*amO2)+K*J*(J-1)  
    # Put principal kirchoff stress in 3x3 matrix    
    TauP33=as_tensor([[TauP1,0,0],[0,TauP2,0],[0,0,TauP3]])
    # Rotate Tau back
    vC = as_tensor([vC1,vC2,vC3])
    Tau = vC*TauP33*vC.T 
    # Cauchy Stress
    sig = (1/J)*Tau
    # Second PK Stress
    return inv(F)*Tau*inv(F.T)

def W_Ogden(F,  muO1=0.64724, alO1=1.3, muO2=0.001177,alO2=5.0, K=20000):
    """
    Strain energy density, 2-term Ogden material
    This needs to be differentiated wrt the principal stretches. 
    """

    C = F.T * F  # Right cauchy-green strain tensor
    eigC1, eigC2, eigC3 = get_eig(C) # Eigenvalues of C


    # Eigenvalues of U (The principal stretches) are the square root of the eigenvalues of C
    lambda1 = ufl.sqrt(eigC1)
    lambda2 = ufl.sqrt(eigC2)
    lambda3 = ufl.sqrt(eigC3)

    J = lambda1*lambda2*lambda3 # Incompressibility parameter
    lambda1_s=lambda1/J**(1/3) # Calculate volumetric-independant principal stretches
    lambda2_s=lambda2/J**(1/3) # Calculate volumetric-independant principal stretches
    lambda3_s=lambda3/J**(1/3) # Calculate volumetric-independant principal stretches

    #     First term
    #W = (muO1/alO1)*(lambda1_s**alO1 + lambda2_s**alO1 + lambda3_s**alO1 - 3) + (muO2/alO2)*(lambda1_s**alO2 + lambda2_s**alO2 + lambda3_s**alO2 - 3) + K*(J-1-ufl.ln(J))
    # See : 19.77.2 Ogden Rubber Model, LS Dyna Theory Manual
    W = (muO1/alO1)*(lambda1_s**alO1 + lambda2_s**alO1 + lambda3_s**alO1 - 3) + (muO2/alO2)*(lambda1_s**alO2 + lambda2_s**alO2 + lambda3_s**alO2 - 3) + 0.5*K*(J-1)**2

    return W


def Piola1_ogden(d):
    """
    First Piola-Kirchhoff Stress (solid problem)
    """
    print("calculated PK1")

    #return F_(d)*S_ogden(d)
    return F_(d)*S_ogden(d)
