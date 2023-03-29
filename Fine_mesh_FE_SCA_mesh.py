#Fine_mesh_compare_with_FE_SCA_mesh
from FEM_utility import *
from Homogenization_utilities import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




Mesh = ReadMesh('2DFINE_notype.inp')
Mesh.Connectivity()

#method 1 directly compare the strain value computed using FE-SCA at the gauss point
#in order to approxiate the strain at surface using fem, the element thickness should be small
U = np.loadtxt('finemesh_U.txt') # finemesh_U
Ux = U[:63001,1]                      #ux
Uy = U[:63001,2]                      #uy
U = np.concatenate((Ux, Uy), axis = 0)
exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(U)

MDICelementlist = MDICElementList(50, 50, 250, 250)
MDICgausspointlist = MDICGaussPointList(MDICelementlist, 250 , 250)

exxMDIC, eyyMDIC, exyMDIC = DIC2MDICstrain(MDICgausspointlist, exxgp, eyygp, exygp)

#import the strain from the FE-SCA simulation at the single gauss point for each element: note this case the gauss point is not at the surface
E_FESCA= np.loadtxt('FE_SCA_strain_GP.txt') # finemesh_U
E_FESCA = E_FESCA[:,2]
print('The L2 norm error for method 1 is', np.linalg.norm(E_FESCA - exxMDIC))  

#method 2
#import 2d mesh for FE-SCA and compute the 2d strain at the gauss point (this time on the surface)

Mesh2 = ReadMesh('2DFESCA_notype.inp')
Mesh2.Connectivity()

U_FESCA = np.loadtxt('fe_sca_U.txt') # finemesh_U
Ux_FESCA = U_FESCA[:36,1]                      #ux
Uy_FESCA = U_FESCA[:36,2]                      #uy
U_FESCA = np.concatenate((Ux_FESCA, Uy_FESCA), axis = 0)
exxgp_FESCA, eyygp_FESCA, exygp_FESCA = Mesh2.StrainAtGaussPoint(U_FESCA)

MDICelementlist = MDICElementList(1, 1, 5, 5) #average the strain at four gauss points to just 1 single gauss point for easier comparison
MDICgausspointlist = MDICGaussPointList(MDICelementlist, 5 , 5)

exxFESCA, eyyFESCA, exyFESCA = DIC2MDICstrain(MDICgausspointlist, exxgp_FESCA, eyygp_FESCA, exygp_FESCA)
print('The L2 norm error for method 2 is', np.linalg.norm(exxMDIC - exxFESCA))  

x = np.arange(0.05, 0.46, 0.1)
y = np.arange(0.05, 0.46, 0.1)
X,Y = np.meshgrid(x,y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.plot_surface(X, Y, np.reshape(exxMDIC + 1, (5,5)), cmap=cm.coolwarm, linewidth=0, alpha = 0.4)
ax.scatter(X, Y, np.reshape(exxFESCA + 1, (5,5)))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Deformation gradient')

plt.show()
print('Hello')