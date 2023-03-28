#Functions for homogenization purposes
#get the ID of all the DIC elements in each MDIC elements
#number of rows: number of MDIC elements
#columns: the ID of all the DIC elements in that element
#layout and numbering of the gauss points

#################
# 12# 13# 14# 15# 
#################
# 8 # 9 # 10 #11# 
#################
# 4 # 5 # 6 # 7 # 
#################
# 0 # 1 # 2 # 3 # 
#################
import numpy as np
def MDICElementList(nx, ny, Nx, Ny):
    num_MDIC_ele_x = int(Nx/nx)
    num_MDIC_ele_y = int(Ny/ny)
    MDIC_ele_list = []
    for j in range(num_MDIC_ele_y):
        for i in range(num_MDIC_ele_x):
            MDIC_ID = i + j * num_MDIC_ele_y
            MDIC_ele_list.append([])
            MDIC_ele = []
            for k in range(ny):
                MDIC_ele = np.append(MDIC_ele, np.arange(j * (nx * ny * num_MDIC_ele_x) + k * Ny + i * nx, j * (nx * ny * num_MDIC_ele_x) + k * Ny + (i+1) * nx, 1, dtype = int))
            MDIC_ele = MDIC_ele.astype(int)
            MDIC_ele_list[MDIC_ID] = list(MDIC_ele)
    return MDIC_ele_list



def MDICGaussPointList(MDIC_ele_list, Nx, Ny):
#get the ID of all the gauss points in each MDIC elements
#number of rows: number of MDIC elements
#columns: the ID of all the gauss points in that element
#layout and numbering of the gauss points
#############
#10 14#11 15#
# 2  6# 3  7#
#############
# 8 12# 9 13#
# 0  4# 1  5#
#############
    MDICGaussPointList = []
    num_DIC_ele = Nx*Ny
    for i in range(len(MDIC_ele_list)):
        MDICGaussPointList.append([])
        GP = []
        for j in MDIC_ele_list[i]:
            GP = np.append(GP , np.array([j, j + num_DIC_ele, j + 2 * num_DIC_ele, j + 3 * num_DIC_ele], dtype = int))
        GP = GP.astype(int)
        
        MDICGaussPointList[i] = list(GP)
    
    return MDICGaussPointList


Mele = MDICElementList(2, 2, 4, 4)
MGP = MDICGaussPointList(Mele, 4, 4)
print(Mele)
