import os, sys, shutil
from dolfin import *

comm = MPI.comm_world
nprocs = comm.Get_size()
if (nprocs>1):
    print('Support serial computation only!')
    sys.exit()


exists = os.path.isfile('DmriFemLib.py')
isupdate = False
if (exists==False or isupdate==True):
    if isupdate==True:
        os.system("rm DmriFemLib.py")
    print("Load pre-defined functions from GitHub")
    os.system("wget --quiet https://raw.githubusercontent.com/rochishnu00/DMRI_rochi/master/DmriFemLib.py")
from DmriFemLib import *

mesh_name="CirclesInSquare"
os.system('gmsh -3 '+mesh_name+'.geo -o '+mesh_name+'.msh')
os.system('dolfin-convert '+mesh_name+'.msh '+mesh_name+'.xml')
mymesh = Mesh(mesh_name+".xml");
GetPartitionMarkers(mesh_name+".msh", "pmk_"+mesh_name+".xml")
partition_marker = MeshFunction("size_t", mymesh, mymesh.topology().dim())
phase, partion_list = CreatePhaseFunc(mymesh, [], [], partition_marker)   

CheckAndCorrectPeriodicity(mymesh, 0, 1e-6)
CheckAndCorrectPeriodicity(mymesh, 1, 1e-6)
CheckAndCorrectPeriodicity(mymesh, 2, 1e-6)

D0_array=[1.6642e-3,0.5447e-3]
V_DG = FunctionSpace(mymesh, 'DG', 0)
dofmap_DG = V_DG.dofmap()

d00 = Function(V_DG); d01 = Function(V_DG); d02 = Function(V_DG)  
d10 = Function(V_DG); d11 = Function(V_DG); d12 = Function(V_DG)
d20 = Function(V_DG); d21 = Function(V_DG); d22 = Function(V_DG)
T2  = Function(V_DG);
disc_ic = Function(V_DG);

IC_array = [1, 1]
T2_array = [1e6, 1e6]
for cell in cells(mymesh):
      cell_dof = dofmap_DG.cell_dofs(cell.index())
      cmk = partition_marker[cell.index()]
      T2.vector()[cell_dof]      = T2_array[cmk];
      disc_ic.vector()[cell_dof] = IC_array[cmk]; 
      d00.vector()[cell_dof]     = D0_array[cmk]; 
      d11.vector()[cell_dof]     = D0_array[cmk]; 
      d22.vector()[cell_dof]     = D0_array[cmk];

ofile = 'files.h5';
for i in range(0, len(sys.argv)):
      arg = sys.argv[i];
      if arg=='-o':
            ofile = sys.argv[i+1];

filename, file_extension = os.path.splitext(ofile)

ofile = filename+'.h5'
f = HDF5File(mymesh.mpi_comm(), ofile, 'w')
f.write(mymesh, 'mesh');  f.write(T2, 'T2'); f.write(disc_ic, 'ic'); f.write(phase, 'phase');
f.write(d00, 'd00'); f.write(d01, 'd01'); f.write(d02, 'd02')
f.write(d10, 'd10'); f.write(d11, 'd11'); f.write(d12, 'd12')
f.write(d20, 'd20'); f.write(d21, 'd21'); f.write(d22, 'd22')

print("Write to ", ofile)

 
