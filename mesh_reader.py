from mesh_fun import *
import numpy as np
# Pixel = ReadMesh('Example1.inp')
# Pixel.Connectivity()

# DICmesh = ReadMesh('Example_coarse.inp')
# DICmesh.Connectivity()

# MDICmesh = ReadMesh('Example_FEM.inp')
# MDICmesh.Connectivity()
Pixel = ReadMesh('2DFINE.inp')
Pixel.Connectivity()

DICmesh = ReadMesh('2DDIC.inp')
DICmesh.Connectivity()

MDICmesh = ReadMesh('2DFEM.inp')
MDICmesh.Connectivity()



U = np.loadtxt('U1.txt') 
U = U[:,1]                      #pixel mesh
UDIC = []
num = int(np.sqrt(Pixel.ndof/2))
for i in range(0, num, 5):
    for j in range(0, num, 5):
        UDIC.append(U[i * num + j])   
Uy = U[num * num : len(U)]
for i in range(0, num, 5):
    for j in range(0, num, 5):
        UDIC.append(Uy[i * num + j]) 
UDIC = np.array(UDIC)

num = int(np.sqrt(len(UDIC)/2))
MUDIC = []
for i in range(0, num, 10):
    for j in range(0, num, 10):
        MUDIC.append(UDIC[i * num + j])   
Uy = U[num * num : len(U)]
for i in range(0, num, 10):
    for j in range(0, num, 10):
        MUDIC.append(Uy[i * num + j]) 
MUDIC = np.array(MUDIC)
#m.Plot()
#m.PlotContourDispl(U)

Pixel.PlotContourStrain(U)
plt.show()

DICmesh.PlotContourStrain(UDIC)
plt.show()


MDICmesh.PlotContourStrain(MUDIC)
plt.show()


#m.StrainAtNodes(self, U):  compute strain at the nodes
#m.StrainAtGP(self, U):  compute strain at the points
#PlotContourDispl(self, U=None, n=None, s=1.0, stype='comp', newfig=True, **kwargs):
        # """
        # Plots the displacement field using Matplotlib Library.        

        # Parameters
        # ----------
        # U : 1D NUMPY.ARRAY
        #     displacement dof vector
        # n : NUMPY.ARRAY, optional
        #     Coordinate of the nodes. The default is None, which corresponds
        #     to using self.n instead.
        # s : FLOAT, optional
        #     Deformation scale factor. The default is 1.0.
        # stype : STRING, optional
        #     'comp' > plots the 3 components of the strain field
        #     'mag' > plots the 'VonMises' equivalent strain
        #      The default is 'comp'.
        # newfig : BOOL
        #     if TRUE plot in a new figure (default)
        # **kwargs : TYPE
        #     DESCRIPTION.

        # Returns
        # -------
        # None.

        # """