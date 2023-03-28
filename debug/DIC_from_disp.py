from FEM_utility import *
import numpy as np
Mesh = ReadMesh('SAMPLE_2D.inp', dim=2)
Mesh.Connectivity()



U = np.loadtxt('Sample_U.txt') 
U = U[:,1]                      #pixel mesh


exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(U)

print('Hello')
