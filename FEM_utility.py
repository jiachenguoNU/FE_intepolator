import numpy as np
import meshio

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






def StrainAtNodes(self, U):
    nnodes = self.ndof // self.dim
    if not hasattr(self, "dphixdx"):
        m = self.Copy()
        m.GaussIntegration()
    else:
        m = self
    if self.dim == 2:
        exxgp = m.dphixdx @ U
        eyygp = m.dphiydy @ U
        exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
        EpsXX = np.zeros(nnodes)
        EpsYY = np.zeros(nnodes)
        EpsXY = np.zeros(nnodes)
        phi = m.phix[:,:nnodes]
        w = np.array(np.sum(phi, axis=0))[0]
        phi = phi @ sp.sparse.diags(1/w)
        EpsXX = phi.T @ exxgp
        EpsYY = phi.T @ eyygp
        EpsXY = phi.T @ exygp
        return EpsXX, EpsYY, EpsXY
    else: #dim 3
        exxgp = m.dphixdx @ U
        eyygp = m.dphiydy @ U
        ezzgp = m.dphizdz @ U
        exygp = 0.5 * m.dphixdy @ U + 0.5 * m.dphiydx @ U
        exzgp = 0.5 * m.dphixdz @ U + 0.5 * m.dphizdx @ U
        eyzgp = 0.5 * m.dphiydz @ U + 0.5 * m.dphizdy @ U
        EpsXX = np.zeros(nnodes)
        EpsYY = np.zeros(nnodes)
        EpsZZ = np.zeros(nnodes)
        EpsXY = np.zeros(nnodes)
        EpsXZ = np.zeros(nnodes)
        EpsYZ = np.zeros(nnodes)
        phi = m.phix[:,:nnodes]
        w = np.array(np.sum(phi, axis=0))[0]
        phi = phi @ sp.sparse.diags(1/w)
        EpsXX = phi.T @ exxgp
        EpsYY = phi.T @ eyygp
        EpsZZ = phi.T @ ezzgp
        EpsXY = phi.T @ exygp
        EpsXZ = phi.T @ exzgp
        EpsYZ = phi.T @ eyzgp
        return EpsXX, EpsYY, EpsZZ, EpsXY, EpsXZ, EpsYZ