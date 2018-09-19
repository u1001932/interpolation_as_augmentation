from PIL import Image
import numpy as np


cat_img = Image.open('automobile8.png')
dog_img = Image.open('airplane4.png')
cat_array = np.array(cat_img)
dog_array = np.array(dog_img)
print("DOG: ", dog_array.shape)
print("CAT: ", cat_array.shape)

#new_array = (dog_array*[0.8,0.8,0.8] + cat_array*[0.2,0.2,0.2])
new_array = np.empty([32,32,3])
for x in range(32):
  for y in range(32):
    for z in range(3):
	#a = np.random.beta(3,3)
        new_array[x,y,z] = 0.2*dog_array[x,y,z] + 0.8*cat_array[x,y,z]
	dog_array[x,y,z] = 0.5*dog_array[x,y,z] + 0.5*cat_array[x,y,z] #a*dog_array[x,y,z] + (1-a)*cat_array[x,y,z]

print(new_array.astype(int))
print(dog_array)

img = Image.fromarray(dog_array,'RGB')
img.save('car-plane.png')
