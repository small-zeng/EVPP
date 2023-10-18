import numpy as np

object_center = [0.0,1.0,0.0] 
pos = [0.69,0.3,-2.84]
dx,dy,dz =  object_center[0]-pos[0],object_center[1]-pos[1],object_center[2]-pos[2]
u = np.arctan2(-dy,np.sqrt(dx**2+dz**2)) 
v = np.arctan2(dx,dz)
print(u,v)
print(u/np.pi*180.,v/np.pi*180.)



# Create your tests here.
