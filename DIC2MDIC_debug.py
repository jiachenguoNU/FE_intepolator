from FEM_utility import *
from Homogenization_utilities import *
import numpy as np

Mesh = ReadMesh('SAMPLE_2D.inp')
Mesh.Connectivity()



U = np.loadtxt('Sample_U.txt') 
U = U[:,1]                      #pixel mesh

exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(U)

MDICelementlist = MDICElementList(2, 2, 4, 4)
MDICgausspointlist = MDICGaussPointList(MDICelementlist, 4 , 4)

exxMDIC, eyyMDIC, exyMDIC = DIC2MDICstrain(MDICgausspointlist, exxgp, eyygp, exygp)

print(exxMDIC)


