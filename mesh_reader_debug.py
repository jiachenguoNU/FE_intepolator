from mesh_fun import *
from Homogenization_utilities import *
import numpy as np
import matplotlib.pyplot as plt

Mesh = ReadMesh('2DFINE.inp')
Mesh.Connectivity()



U = np.loadtxt('finemesh_U.txt') 
U = U[:,1] 
# exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(U)

print('Hello')
Mesh.PlotContourStrain(U)
plt.show()



