from mesh_fun import *
m = ReadMesh('Example1.inp')
m.Plot()

#m.StrainAtNodes(self, U):  compute strain at the nodes
#m.StrainAtGP(self, U):  compute strain at the points
#PlotContourDispl(self, U=None, n=None, s=1.0, stype='comp', newfig=True, **kwargs):
        # """
        # Plots the displacement field using Matplotlib Library.        

        # Parameters
        # ----------
        # U : 1D NUMPY.ARRAY
        #     displacement dof vector
        # n : NUMPY.ARRAY, optional
        #     Coordinate of the nodes. The default is None, which corresponds
        #     to using self.n instead.
        # s : FLOAT, optional
        #     Deformation scale factor. The default is 1.0.
        # stype : STRING, optional
        #     'comp' > plots the 3 components of the strain field
        #     'mag' > plots the 'VonMises' equivalent strain
        #      The default is 'comp'.
        # newfig : BOOL
        #     if TRUE plot in a new figure (default)
        # **kwargs : TYPE
        #     DESCRIPTION.

        # Returns
        # -------
        # None.

        # """