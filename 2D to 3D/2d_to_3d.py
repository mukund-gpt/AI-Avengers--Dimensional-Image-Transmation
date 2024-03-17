import numpy as np
from stl import mesh
#importing the image file
from PIL import Image
import matplotlib.pyplot as plt
im=Image.open("OIP.jpg")
plt.imshow(im)
#converting image into grey scale
grey_img=Image.open("OIP.jpg").convert("L")
plt.imshow(grey_img)
#creating mesh with triangle
nrows=grey_img.height
ncols=grey_img.width
imagenp=np.array(grey_img)
# print("max=",imagenp.max())
print(imagenp)
max_height=25
maxpix=imagenp.max()

print("shape:",imagenp.shape)
vertices=np.zeros((nrows,ncols,3))
for x in range(0,ncols):
  for y in range(0,nrows):
    pixelinesity=imagenp[y][x]
    z=(pixelinesity*max_height)/maxpix
    vertices[y][x]=(x,y,z)
faces=[]

for x in range(0,ncols-1):
  for y in range(0,nrows-1):
   
    vertice1=vertices[y][x]
    vertice2=vertices[y+1][x]
    vertice3=vertices[y+1][x+1]
    face1=np.array((vertice1,vertice2,vertice3))

    vertice1=vertices[y][x]
    vertice2=vertices[y][x+1]
    vertice3=vertices[y+1][x+1]
    face2=np.array((vertice1,vertice2,vertice3))

    faces.append(face1)
    faces.append(face2)
facesNp=np.array(faces)
#create the mesh
surface=mesh.Mesh(np.zeros(facesNp.shape[0],dtype=mesh.Mesh.dtype))
for i,f in enumerate(faces):
  for j in range(3):
    surface.vectors[i][j]=facesNp[i][j]
#write the mesh to file
surface.save('surface1.stl')
print(surface)

