# BVP
# 
# laplace equation
# two disks with radius 5 and 1, with centres at (0, 0) and (2, 0)
# dirichlet on outer_disk and neumann on inner_disk


import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import pyvista as pv
from dolfinx import mesh, fem, plot

# #1 generate concentric balls domain of radius 5 and 1   drichlet drichlet 

inner_radius = 1.0
outer_radius = 5.0

import gmsh
from dolfinx.io import gmshio

def get_domain(inner_radius: float, outer_radius: float, filename: str = None):
    """
    inner_radius: float, 
    outer_radius: float
    """

    gmsh.initialize()
    model = gmsh.model()
    model.add("domain")
    model.setCurrent("domain")

    outer_disk = model.occ.addDisk(0, 0, 0, outer_radius,outer_radius, tag=1)
    inner_disk = model.occ.addDisk(2, 0, 0, inner_radius,inner_radius, tag=2)
    shell_dims_tags, _ = model.occ.cut( #perform boolean cut
        [(2, outer_disk)], #target disk
        [(2, inner_disk)]  # tool disk
    )
    model.occ.synchronize()
    model.addPhysicalGroup(2, [shell_dims_tags[0][1]], tag = 1) #remaining area
    
    #set mesh options and generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.1)
    model.geo.synchronize()
    model.mesh.generate(2)

    #get domain from mesh
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(model=model, comm=MPI.COMM_WORLD, rank = 0, gdim = 2)
    gmsh.finalize()
    return domain, cell_markers, facet_markers

# For plotting domain!
from dolfinx import plot
import pyvista

def plot_domain(domain):

    pyvista.start_xvfb()
    gdim = domain.topology.dim #dimensions of domain!

    domain.topology.create_connectivity(gdim, gdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, gdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        figure = plotter.screenshot("fundamentals_mesh.png")

domain, cell_markers, facet_markers = get_domain(inner_radius, outer_radius)
# plot_domain(domain)

# #2 Setup function space and apply boundary conditions
def setup_fe_space(domain):
    """Create function space and variational forms. Defining bilinear and linear forms."""
    V = fem.functionspace(domain, ("Lagrange", 1)) #set function space

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
    m = fem.form(ufl.inner(u, v)*ufl.dx)

    return V, a, m

V, a, m = setup_fe_space(domain)

# preparing boundary conditions
# Drichlet BC u = 0
from petsc4py import PETSc
def on_boundary(x):
    return np.isclose(np.sqrt((x[0])**2 + x[1]**2), outer_radius)

def get_bcs(V):
    boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)

    return bc

# #3 assemble matrices and apply boundary conditions
import dolfinx.fem.petsc

def assemble_and_apply_bc(V, a, m):
    #get bcs
    bc = get_bcs(V)

    #  Assemble matrices A and M
    A = fem.petsc.assemble_matrix(a, bcs=[bc]) #bc only on stiffness matrix
    A.assemble()
    M = fem.petsc.assemble_matrix(m) # mass matrix
    M.assemble()

    return A, M

A, M = assemble_and_apply_bc(V, a, m)

print(A.norm())
print(M.norm())

# setup eigenvalue solver
def get_eigenvalues(A, M, n_eigen):

    eigensolver = SLEPc.EPS().create()
    eigensolver.setOperators(A, M)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    # eigensolver.setDimensions(nev=n_eigen, ncv=2*n_eigen)  # Larger subspace
    eigensolver.setDimensions(nev=n_eigen)  # Larger subspace
    # eigensolver.setTolerances(tol=1e-8, max_it=1000)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

    eigensolver.setFromOptions()
    eigensolver.solve()
    n_conv = eigensolver.getConverged()

    if MPI.COMM_WORLD.rank == 0:
        print(f"Number of converged eigenvalues: {n_conv}")
        for i in range(n_conv):
            eigval = eigensolver.getEigenvalue(i)
            print(f"Eigenvalue {i+1}: {eigval:.4f}")

    return eigensolver, n_conv


n_eigen = 3
eigensolver, n_conv = get_eigenvalues(A, M, n_eigen=n_eigen)

import matplotlib.pyplot as plt
import numpy as np
import dolfinx.plot
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, compute_midpoints, create_unit_cube, create_unit_square, meshtags

def plot_eigenfunctions(domain, V, eigensolver, A, nev_to_plot):
    """
    Plot the first few eigenfunctions for any given domain using matplotlib.
    Parameters:
        domain: dolfinx.mesh.Mesh
        V: dolfinx.fem.FunctionSpace
        eigensolver: SLEPc.EPS
        A: PETSc.Mat (needed for getVecs)
        n_conv: int (number of converged eigenpairs)
        nev_to_plot: int (number of eigenfunctions to plot)
    """
    if nev_to_plot == 0:
        return
    
    if MPI.COMM_WORLD.rank != 0:
        return

    topology, cell_types, geometry = plot.vtk_mesh(V)
    points = domain.geometry.x
    for i in range(nev_to_plot):
        eigval = eigensolver.getEigenvalue(i)
        error = eigensolver.computeError(i)

        print(f"Eigenvalue {i}: {eigval:.4f} (Error: {error:.4e})")

        r, _ = A.getVecs()
        eigensolver.getEigenvector(i, r)

        uh = fem.Function(V)
        uh.x.petsc_vec.setArray(r.array)
        uh.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Create PyVista grid
        values = uh.x.array.real
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["u"] = values
        
        # The function "u" is set as the active scalar for the mesh, and
        # warp in z-direction is set
        grid.set_active_scalars("u")
        warped = grid.warp_by_scalar('u', factor=25)
        
        # Define the number of contour levels
        contour_levels = 30
        # Generate contours using the scalar data
        contours = grid.contour(contour_levels, scalars="u")
        
        # A plotting window is created with two sub-plots, one of the scalar
        # values and the other of the mesh is warped by the scalar values in
        # z-direction
        subplotter = pv.Plotter(shape=(1, 2))
        subplotter.subplot(0, 0)
        subplotter.add_text("Eigenfunction", font_size=14, color="black", position="upper_edge")
        subplotter.add_mesh(grid, show_scalar_bar=True, cmap="jet")
        subplotter.add_mesh(contours, color="blue", line_width=1)
        subplotter.view_xy()

        contours_warp = warped.contour(contour_levels, scalars="u")
        
        subplotter.subplot(0, 1)
        subplotter.add_text("3-D visualization", position="upper_edge", font_size=14, color="black")
        
        subplotter.add_mesh(warped, show_scalar_bar=True, cmap="jet")
        subplotter.add_mesh(contours_warp, color="blue", line_width=1)

        subplotter.show()

plot_eigenfunctions(domain, V, eigensolver, A, n_conv)








