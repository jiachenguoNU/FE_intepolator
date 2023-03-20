# from FEM_utility import *
from mesh_fun import *
import numpy as np
Mesh = ReadMesh('SAMPLE_2D.inp', dim=2)
Mesh.Connectivity()



U = np.loadtxt('Sample_U.txt') 
U = U[:,1]                      #pixel mesh


# exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(U)

print('Hello')
Mesh.PlotContourStrain(U)
plt.show()



