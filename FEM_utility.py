import numpy as np
import meshio
import scipy as sp

def ElTypes():
    """
    Returns a dictionnary of GMSH element types which some of them are used in the library.
    """
    return {
      1: "2-node line.",
      2: "3-node triangle.",
      3: "4-node quadrangle.",
      4: "4-node tetrahedron.",
      5: "8-node hexahedron.",
      6: "6-node prism.",
      7: "5-node pyramid.",
      8: "3-node second order line (2 nodes associated with the vertices and 1 with the edge).",
      9: "6-node second order triangle (3 nodes associated with the vertices and 3 with the edges).",
      10: "9-node second order quadrangle (4 nodes associated with the vertices, 4 with the edges and 1 with the face).",
      11: "10-node second order tetrahedron (4 nodes associated with the vertices and 6 with the edges).",
      12: "27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume).",
      13: "18-node second order prism (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces).",
      14: "14-node second order pyramid (5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face).",
      15: "1-node point.",
      16: "8-node second order quadrangle (4 nodes associated with the vertices and 4 with the edges).",
      17: "20-node second order hexahedron (8 nodes associated with the vertices and 12 with the edges).",
      18: "15-node second order prism (6 nodes associated with the vertices and 9 with the edges).",
      19: "13-node second order pyramid (5 nodes associated with the vertices and 8 with the edges).",
      20: "9-node third order incomplete triangle (3 nodes associated with the vertices, 6 with the edges)",
      21: "10-node third order triangle (3 nodes associated with the vertices, 6 with the edges, 1 with the face)",
      22: "12-node fourth order incomplete triangle (3 nodes associated with the vertices, 9 with the edges)",
      23: "15-node fourth order triangle (3 nodes associated with the vertices, 9 with the edges, 3 with the face)",
      24: "15-node fifth order incomplete triangle (3 nodes associated with the vertices, 12 with the edges)",
      25: "21-node fifth order complete triangle (3 nodes associated with the vertices, 12 with the edges, 6 with the face)",
      26: "4-node third order edge (2 nodes associated with the vertices, 2 internal to the edge)",
      27: "5-node fourth order edge (2 nodes associated with the vertices, 3 internal to the edge)",
      28: "6-node fifth order edge (2 nodes associated with the vertices, 4 internal to the edge)",
      29: "20-node third order tetrahedron (4 nodes associated with the vertices, 12 with the edges, 4 with the faces)",
      30: "35-node fourth order tetrahedron (4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume)",
      31: "56-node fifth order tetrahedron (4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume)",
      92: "64-node third order hexahedron (8 nodes associated with the vertices, 24 with the edges, 24 with the faces, 8 in the volume)",
      93: "125-node fourth order hexahedron (8 nodes associated with the vertices, 36 with the edges, 54 with the faces, 27 in the volume)",
      }

eltype_n2s = {1: "line",
        2: "triangle",
        3: "quad",
        4: "tetra",
        5: "hexahedron",
        6: "wedge",
        7: "pyramid",
        8: "line3",
        9: "triangle6",
        10: "quad9",
        11: "tetra10",
        12: "hexahedron27",
        14: "pyramid14",
        15: "vertex",
        16: "quad8",
        17: "hexahedron20",
        18: "wedge15",
        19: "pyramid13"}

eltype_s2n = {}
for jn in eltype_n2s.keys():
    eltype_s2n[eltype_n2s[jn]] = jn


def ShapeFunctions(eltype):
    """For any type of 2D elements, gives the quadrature rule and
    the shape functions and their derivative"""
    if eltype == 1:
        """
        #############
            seg2
        #############
        """
        def N(x):
            return np.concatenate(
                (0.5 * (1 - x), 0.5*(x + 1))).reshape((2,len(x))).T
        
        def dN_xi(x):
            return np.concatenate(
                (-0.5 + 0 * x, 0.5 + 0 * x)).reshape((2,len(x))).T

        # def dN_eta(x):
        #     return False
        # 3GP
        # xg = np.sqrt(3/5) * np.array([-1, 0, 1])
        # wg = np.array([5., 8., 5.])/9
        # 2GP
        # xg = np.sqrt(3)/3 * np.array([-1, 1])
        # wg = np.array([1., 1.])
        # 1GP
        xg = np.array([0.])
        wg = np.array([2.])
        return xg, wg, N, dN_xi
    elif eltype == 3:
        """
        #############
            qua4 in the parent coor
        #############
        """
        def N(x, y):
            return 0.25 * np.concatenate(((1 - x) * (1 - y),
                                 (1 + x) * (1 - y),
                                 (1 + x) * (1 + y),
                                 (1 - x) * (1 + y))).reshape((4,len(x))).T 

        def dN_xi(x, y):
            return 0.25 * np.concatenate(
                (y - 1, 1 - y, 1 + y, -1 - y)).reshape((4,len(x))).T 

        def dN_eta(x, y):
            return 0.25 * np.concatenate(
                (x - 1, -1 - x, 1 + x, 1 - x)).reshape((4,len(x))).T 

        xg = np.sqrt(3) / 3 * np.array([-1, 1, -1, 1])
        yg = np.sqrt(3) / 3 * np.array([-1, -1, 1, 1])
        wg = np.ones(4)
        return xg, yg, wg, N, dN_xi, dN_eta
    


def ReadMesh(fn, dim=2):
    mesh = meshio.read(fn)
    # file_format="stl",  # optional if filename is a path; inferred from extension
    if mesh.points.shape[1] > dim: # too much node coordinate
        # Remove coordinate with minimal std.
        rmdim = np.argmin(np.std(mesh.points,axis=0))
        n = np.delete(mesh.points, rmdim, 1)
    elif mesh.points.shape[1] < dim: # not enough node coordinates
        n = np.hstack((mesh.points, np.zeros((len(mesh.points),1))))
    else :
        n = mesh.points
    e = dict()
    for et in mesh.cells_dict.keys():
        e[eltype_s2n[et]] = mesh.cells_dict[et]
    m = Mesh(e, n, dim)
    m.point_data = mesh.point_data
    m.cell_data = mesh.cell_data
    return m

class Mesh:
    def __init__(self, e, n, dim=2):
        """Contructor from elems and node arrays"""
        self.e = e
        self.n = n
        self.conn = []
        self.ndof = []
        self.npg = []
        self.pgx = []
        self.pgy = []
        self.phix = None
        self.phiy = None
        self.wdetJ = []
        self.dim = dim

    def Copy(self):
        m = Mesh(self.e.copy(), self.n.copy())
        m.conn = self.conn.copy()
        m.ndof = self.ndof
        m.dim = self.dim
        m.npg = self.npg
        m.pgx = self.pgx.copy()
        m.pgy = self.pgy.copy()
        m.phix = self.phix
        m.phiy = self.phiy
        m.wdetJ = self.wdetJ.copy()
        return m
    
    
    def Connectivity(self):
        print("Connectivity.")
        """ Computes connectivity """
        used_nodes = np.zeros(0, dtype=int)
        for je in self.e.keys():
            used_nodes = np.unique(np.append(used_nodes, self.e[je].ravel()))
        nn = len(used_nodes)
        self.conn = -np.ones(self.n.shape[0], dtype=int)
        self.conn[used_nodes] = np.arange(nn)
        if self.dim == 2:
            self.conn = np.c_[self.conn, self.conn + nn * (self.conn >= 0)]
        else:
            self.conn = np.c_[
                self.conn,
                self.conn + nn * (self.conn >= 0),
                self.conn + 2 * nn * (self.conn >= 0),
            ]
        self.ndof = nn * self.dim





    def __GaussIntegElem(self, e, et):
        # parent element
        xg, yg, wg, N, Ndx, Ndy = ShapeFunctions(et)
        phi = N(xg, yg) #shape function value at gauss point
        dN_xi = Ndx(xg, yg)
        dN_eta = Ndy(xg, yg)
        # elements
        ne = len(e)  # nb (number) of elements
        nfun = phi.shape[1]  # nb of shape fun per element
        npg = len(xg)  # nb of gauss point per element
        nzv = nfun * npg * ne  # nb of non zero values in dphixdx
        wdetJ = np.zeros(npg * ne)
        row = np.zeros(nzv, dtype=int)
        col = np.zeros(nzv, dtype=int)
        val = np.zeros(nzv)
        valx = np.zeros(nzv)
        valy = np.zeros(nzv)
        repdof = self.conn[e, 0]
        xn = self.n[e, 0]
        yn = self.n[e, 1]
        for i in range(len(xg)): #what's dr and ds
            dxdr = xn @ dN_xi[i, :] #dxdksi
            dydr = yn @ dN_xi[i, :] #dydksi
            dxds = xn @ dN_eta[i, :]
            dyds = yn @ dN_eta[i, :]
            detJ = dxdr * dyds - dxds * dydr #Jacobian @each gauss point
            wdetJ[np.arange(ne) + i * ne] = abs(detJ) * wg[i]
            dphidx = (dyds / detJ)[:, np.newaxis] * dN_xi[i, :] + (-dydr / detJ)[
                :, np.newaxis] * dN_eta[i, :]
            dphidy = (-dxds / detJ)[:, np.newaxis] * dN_xi[i, :] + (dxdr / detJ)[
                :, np.newaxis] * dN_eta[i, :]
            repnzv = np.arange(ne * nfun) + i * ne * nfun #column number for different item for each gauss point
            col[repnzv] = repdof.ravel()
            row[repnzv] = np.tile(np.arange(ne) + i * ne, [nfun, 1]).T.ravel()
            val[repnzv] = np.tile(phi[i, :], [ne, 1]).ravel()
            valx[repnzv] = dphidx.ravel()
            valy[repnzv] = dphidy.ravel()
        return col, row, val, valx, valy, wdetJ


    def ShapeFunctionGaussPoint(self):
        """Builds a Gauss integration scheme"""
        print('Shape function at Gauss Points has been initialized.')
        self.wdetJ = np.array([])
        col = np.array([])
        row = np.array([])
        val = np.array([])
        valx = np.array([])
        valy = np.array([])
        npg = 0
        for je in self.e.keys():
            colj, rowj, valj, valxj, valyj, wdetJj = self.__GaussIntegElem(self.e[je], je)
            col = np.append(col, colj)
            row = np.append(row, rowj + npg)
            val = np.append(val, valj)
            valx = np.append(valx, valxj)
            valy = np.append(valy, valyj)
            self.wdetJ = np.append(self.wdetJ, wdetJj)
            npg += len(wdetJj)
        self.npg = len(self.wdetJ)
        colx = col + 0 * self.ndof // self.dim
        coly = col + 1 * self.ndof // self.dim
        self.phix = sp.sparse.csr_matrix(
            (val, (row, colx)), shape=(self.npg, self.ndof))
        self.phiy = sp.sparse.csr_matrix(
            (val, (row, coly)), shape=(self.npg, self.ndof))
        self.dphixdx = sp.sparse.csr_matrix(
            (valx, (row, colx)), shape=(self.npg, self.ndof))
        self.dphixdy = sp.sparse.csr_matrix(
            (valy, (row, colx)), shape=(self.npg, self.ndof))
        self.dphiydx = sp.sparse.csr_matrix(
            (valx, (row, coly)), shape=(self.npg, self.ndof))
        self.dphiydy = sp.sparse.csr_matrix(
            (valy, (row, coly)), shape=(self.npg, self.ndof))
        rep, = np.where(self.conn[:, 0] >= 0)
        qx = np.zeros(self.ndof)
        qx[self.conn[rep, :]] = self.n[rep, :]
        self.pgx = self.phix.dot(qx)
        self.pgy = self.phiy.dot(qx)

    def StrainAtGaussPoint(self, U):
        nnodes = self.ndof // self.dim
        if not hasattr(self, "dphixdx"):
            m = self.Copy()
            m.ShapeFunctionGaussPoint()
        else:
            m = self
        exxgp = m.dphixdx @ U
        eyygp = m.dphiydy @ U
        exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
        return exxgp, eyygp, exygp