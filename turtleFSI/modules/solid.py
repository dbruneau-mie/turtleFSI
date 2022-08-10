# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *

from dolfin import Constant, inner, grad, MPI


def solid_setup(d_, v_, p_, phi, psi, gamma, dx_s, dx_s_id_list, solid_properties, k, theta,
                gravity, mesh, **namespace):

    # DB added gravity in 3d functionality and multi material capability 16/3/21
    #
    #

    """
    ALE formulation (theta-scheme) of the non-linear elastic problem:
    dv/dt - f + div(sigma) = 0   with v = d(d)/dt
    """

    # From the equation defined above we have to include the equation v - d(d)/dt = 0. This
    # ensures both that the variable d and v is well defined in the solid equation, but also
    # that there is continuity of the velocity at the boundary. Since this is imposed weakly
    # we 'make this extra important' by multiplying with a large number delta.

    delta = 1.0E7
    alpha_p = 10e-9

    # Theta scheme constants
    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    F_solid_linear = 0
    F_solid_nonlinear = 0
    for solid_region in range(len(dx_s_id_list)):
        rho_s = solid_properties[solid_region]["rho_s"]

        ## Temporal term and convection 
        F_solid_linear += (rho_s/k * inner(v_["n"] - v_["n-1"], psi)*dx_s[solid_region]
                          + delta * rho_s * (1 / k) * inner(d_["n"] - d_["n-1"], phi) * dx_s[solid_region]
                          - delta * rho_s * inner(theta0 * v_["n"] + theta1 * v_["n-1"], phi) * dx_s[solid_region]) # Maybe we could add viscoelasticity with v["n"] term instead of (1 / k) * inner(d_["n"] - d_["n-1"], phi) 
        
        # Viscoelasticity (rate dependant portion of the stress)
        if "viscoelasticity" in solid_properties[solid_region]:
            if solid_properties[solid_region]["viscoelasticity"] == "Form1": # This version (using velocity directly) seems to work best. 
                F_solid_nonlinear += theta0 * inner(F_(d_["n"])*Svisc_D(v_["n"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
                F_solid_linear += theta1 * inner(F_(d_["n-1"])*Svisc_D(v_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
            elif solid_properties[solid_region]["viscoelasticity"] == "Form2": # This version (using displacements to calculate linearized derivative) doesnt work as well.
                # (1/k) can come outside because all operators are linear. (v_[n]+v_[n-1])/2 = (d_[n]-d_[n-1])/k    where k is the timestep.
                F_solid_nonlinear += (1/k) * inner(F_(d_["n"])*theta0*Svisc_D(d_["n"] - d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
                F_solid_linear += (1/k) * inner(F_(d_["n-1"])*theta1*Svisc_D(d_["n"] - d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
            else:
                if MPI.rank(MPI.comm_world) == 0:
                    print("Invalid/No entry for viscoelasticity, assuming no viscoelasticity.")
        else:
            if MPI.rank(MPI.comm_world) == 0:
                print("No entry for viscoelasticity, assuming no viscoelasticity.")

        # Stress (Note that if viscoelasticity is used, Piola1() is no longer the total stress, it is the non-rate dependant (elastic) component of the stress)
        F_solid_nonlinear += theta0 * inner(Piola1(d_["n"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
        F_solid_linear += theta1 * inner(Piola1(d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]

        F_solid_linear += alpha_p * inner( grad(p_["n"]), grad(gamma) ) * dx_s[solid_region]


        #F_fluid_nonlinear += inner(J_(d_["n"]) * sigma_f_p(p_["n"], d_["n"]) *

        # Gravity
        if gravity is not None and mesh.geometry().dim() == 2:
            F_solid_linear -= inner(Constant((0, -gravity * rho_s)), psi)*dx_s[solid_region] 
        elif gravity is not None and mesh.geometry().dim() == 3:
            F_solid_linear -= inner(Constant((0, -gravity * rho_s,0)), psi)*dx_s[solid_region] 

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
