from FEM_utility import *
import numpy as np

DICmesh = ReadMesh('2DDIC.inp')
DICmesh.Connectivity()

MDICmesh = ReadMesh('2DFEM.inp')
MDICmesh.Connectivity()



U = np.loadtxt('U1.txt') 
U = U[:,1]                      #pixel mesh
UDIC = U
UDIC = np.array(UDIC)

exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(UDIC)


