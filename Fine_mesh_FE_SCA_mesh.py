#Fine_mesh_compare_with_FE_SCA_mesh
from FEM_utility import *
from Homogenization_utilities import *
import numpy as np
import matplotlib.pyplot as plt

Mesh = ReadMesh('2DFINE_notype.inp')
Mesh.Connectivity()



U = np.loadtxt('finemesh_U.txt') 
Ux = U[:63001,1]                      #ux
Uy = U[:63001,2]                      #uy
U = np.concatenate((Ux, Uy), axis = 0)
exxgp, eyygp, exygp = Mesh.StrainAtGaussPoint(U)

MDICelementlist = MDICElementList(50, 50, 250, 250)
MDICgausspointlist = MDICGaussPointList(MDICelementlist, 250 , 250)

exxMDIC, eyyMDIC, exyMDIC = DIC2MDICstrain(MDICgausspointlist, exxgp, eyygp, exygp)

print('Hello')