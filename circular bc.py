# %% [markdown]
# BVP
# 
# laplace equation
# drichlet bc over disk of radius 1 units around center (0, 0)
# least 10 eigenvalues and eigenfunctions

# %%
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import pyvista as pv
from dolfinx import mesh, fem, plot
# import pyvista as pv
# pv.start_xvfb()



# %% [markdown]
# create a domain. We need circle. using gmsh to create domain extenally and loading to fenics!

# %%
import gmsh
gmsh.initialize()

# %%
disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()


# %%
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [disk], 1)


# %%
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)


# %%
from dolfinx.io import gmshio 
domain, cell_markers, facet_markers = gmshio.model_to_mesh(model=gmsh.model, comm=MPI.COMM_WORLD,rank= 0, gdim=gdim)

# %%
domain

# %% [markdown]
# Got disk domain!, create function space
# 

# %%
V = fem.functionspace(domain, ("Lagrange", 1))

# %% [markdown]
# setting the boundary conditions
# as our boundary is circle
# so we'll set the boundary as sqrt(x2+y2) = 1

# %%
def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)

boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)

# %% [markdown]
# as our u = 0 at boundary, we have drichlet condition

# %%
from petsc4py import PETSc
default_scalar_type = PETSc.ScalarType

bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

# %%
#define biliear and linear forms
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
m = fem.form(ufl.inner(u, v) * ufl.dx)

# %%
import dolfinx.fem.petsc
# 4. Assemble matrices A and M
A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
M = fem.petsc.assemble_matrix(m, bcs=[bc])
M.assemble()

# %%
# 5. Solve the generalized eigenvalue problem A u = λ M u
eigensolver = SLEPc.EPS().create()
eigensolver.setOperators(A, M)
# eigensolver.setProblemType(SLEPc.EPS.ProblemType.HEP)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eigensolver.setWhichEigenpairs(2) # set 1 for maximum eigenvalues
eigensolver.setDimensions(nev=10)  # Get 10 eigenvalues
eigensolver.setFromOptions()
eigensolver.solve()

n_conv = eigensolver.getConverged()
if MPI.COMM_WORLD.rank == 0:
    print(f"Number of converged eigenvalues: {n_conv}")


# %%
# 6. Extract and plot first few eigenfunctions
if MPI.COMM_WORLD.rank == 0:
    topology, cell_types, geometry = plot.vtk_mesh(V)
    points = domain.geometry.x
    for i in range(min(n_conv, 10)):
        eigval = eigensolver.getEigenvalue(i)
        print(f"Eigenvalue {i}: {eigval:.4f}")

        r, _ = A.getVecs()
        eigensolver.getEigenvector(i, r)

        uh = fem.Function(V)
        uh.x.petsc_vec.setArray(r.array)
        uh.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Create PyVista grid
        values = uh.x.array.real
        grid = pv.UnstructuredGrid(topology, cell_types, points)
        grid.point_data["u"] = values

        # Plot
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, scalars="u", cmap="viridis")
        plotter.view_xy()
        plotter.add_text(f"Eigenfunction {i}, λ = {eigval:.2f}", font_size=12)
        plotter.show()


# %%



