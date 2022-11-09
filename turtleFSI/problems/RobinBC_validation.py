from dolfin import *
import os
from turtleFSI.problems import *
import numpy as np
from scipy.integrate import odeint

"""
This problem is a validation of the Robin BC implementation in the solid solver.
The solid is a cylinder with a hole in the middle. 
The domain consits of only a solid.
"""

# Set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet normals n('+') within interior boundaries (dS). For 3D mesh the value should be "shared_vertex", for 2D mesh "shared_facet", the default value is "none".
parameters["ghost_mode"] = "shared_vertex" #3D case
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1E6                              # Young modulus (elasticity) [Pa]
    nu_s_val = 0.45                            # Poisson ratio (compressibility)
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # Shear modulus
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    # define and set problem variables values
    default_variables.update(dict(
        T=0.5,                              # Simulation end time
        dt=0.001,                           # Time step size
        theta=0.501,                        # Theta scheme (implicit/explicit time stepping): 0.5 + dt
        atol=1e-7,                          # Absolute tolerance in the Newton solver
        rtol=1e-7,                          # Relative tolerance in the Newton solver
        mesh_file="cylinder",               # Mesh file name
        inlet_id=2,                         # inlet id 
        outlet_id1=3,                       # outlet id
        inlet_outlet_s_id=1011,             # solid inlet and outlet id
        fsi_id=1022,                        # fsi Interface 
        rigid_id=1011,                      # "rigid wall" id for the fluid and mesh problem
        outer_wall_id=1033,                 # outer surface / external id 
        ds_s_id=[1033],                     # ID of solid external wall (where we want to test Robin BC)
        rho_f=1.025E3,                      # Fluid density [kg/m3]
        mu_f=3.5E-3,                        # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0E3,                        # Solid density [kg/m3]
        mu_s=mu_s_val,                      # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,                      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,              # Solid 1rst Lamé coef. [Pa]
        robin_bc = True,                    # Robin BC
        k_s = 1.0E5,                        # elastic response necesary for RobinBC
        c_s = 0,                            # viscoelastic response necesary for RobinBC 
        u_max= 0,                           # max inlet flow velocity value [m/s]
        p_val= 0,                           # inner pressure for initialisation [Pa]
        vel_t_ramp= 0.2,                    # time for velocity ramp 
        p_t_ramp_start = 0.2,               # pressure ramp start time
        p_t_ramp_end = 0.4,                 # pressure ramp end time
        dx_f_id=1,                          # ID of marker in the fluid domain
        dx_s_id=1002,                       # ID of marker in the solid domain
        extrapolation="laplace",            # laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="constant",  # constant, small_constant, volume, volume_change, bc1, bc2
        recompute=30,                       # Number of iterations before recompute Jacobian. 
        recompute_tstep=10,                 # Number of time steps before recompute Jacobian. 
        save_step=1,                        # Save frequency of files for visualisation
        folder="robinbc_validation",        # Folder where the results will be stored
        checkpoint_step=50,                 # checkpoint frequency
        kill_time=100000,                   # in seconds, after this time start dumping checkpoints every timestep
        save_deg=1,                         # Default could be 1. 1 saves the nodal values only while 2 takes full advantage of the mide side nodes available in the P2 solution. P2 for nice visualisations 
        gravity=2.0,                        # Gravitational force [m/s^2] 
        fluid="no_fluid"                    # Do not solve for the fluid
    ))

    return default_variables


def get_mesh_domain_and_boundaries(mesh_file, **namespace):
    #Import mesh file
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "mesh/" + mesh_file + ".h5", "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")
    
    #Set all solid
    domains.set_all(1002)
    
    #Print mesh Volume
    V = assemble(1*dx(mesh))
    if MPI.rank(MPI.comm_world) == 0:
        print("Volume of the mesh: ", V)

    return mesh, domains, boundaries

def create_bcs(**namespace):
    """
    In this problem we use Robin boundary condition which is implemented in the solid.py file.
    Thus, we do not need specify any boundary conditions in this function. 
    """
    return dict(bcs=[])

# TODO: add analytical solution

def _mass_spring_damper_system_ode(x,t):
    pass

def _analytical_solution():
    pass