# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2022 replay file
# Internal Version: 2021_09_15-12.57.30 176069
# Run by jgz1751 on Mon Mar 27 22:02:42 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Abaqus Warning: Unknown keyword (importEnv) in environment file.
#: Abaqus Warning: Unknown keyword (platform) in environment file.
#: Abaqus Warning: Please check spelling of the environment variable names.
#:                 Unknown keyword "keywordname" can be removed using "del keywordname"
#:                 at the end of environment file.
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=402.5234375, 
    height=203.777770996094)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
#: Abaqus Warning: Unknown keyword (importEnv) in environment file.
#: Abaqus Warning: Unknown keyword (platform) in environment file.
#: Abaqus Warning: Please check spelling of the environment variable names.
#:                 Unknown keyword "keywordname" can be removed using "del keywordname"
#:                 at the end of environment file.
o2 = session.openOdb(name='Sample.odb')
#: Model: C:/Users/JGZ1751/OneDrive - Northwestern University/git/FE_intepolator/Sample.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          4
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
