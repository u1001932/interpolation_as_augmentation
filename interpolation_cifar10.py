import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


'''
Returns augmented images AND labels that are linear 
combinations of randomly selected data from the first rows in x_data, 
and y_data. Note y_data must be converted to one-hot. 

:params train_size: number of images to choose from data, augmentation_ratio:
    how many times more augmented images to produce, compared to train_size,
    x_data: np representation of images (cifar10 formate assumed), y_data: class labels

'''
IMAGE_WIDTH=32
IMAGE_HEIGHT=32
CHANNELS = 3
INTERPOLATION_LEVEL=0.4

def interpolate(train_size=20000, augmentation_ratio=5, x_data=None, y_data=None):
    results = []
    labels = []
    if x_data is None:
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    else:
        train_x = x_data
        train_y = y_data
    
    enc = OneHotEncoder()
    train_y = enc.fit_transform(train_y)
    print(train_y.shape)
    
    for n in range(train_size * augmentation_ratio):
        pic1_index, pic2_index = np.random.randint(0,train_size,2)
        pic1 = train_x[pic1_index]
        pic2 = train_x[pic2_index]
        ratio = np.random.uniform(0,INTERPOLATION_LEVEL)
        
        # new_label = ratio * pic1_label + (1-ratio) * pic2_label (elementwise addition)
        new_label = []
        for n in range(train_y.shape[1]):
            new_label.append(ratio * train_y[pic1_index, n] + (1-ratio) * train_y[pic2_index, n])
        labels.append(new_label)
        
        # new_image = ratio * pic1 + (1-ratio) * pic 2 (elemenrwise addition)
        new_image = np.zeros([IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])
        for z in range(CHANNELS):
            for y in range(IMAGE_HEIGHT):
                for x in range(IMAGE_WIDTH):
                    new_image[x,y,z] = np.add(ratio * pic1[x,y,z], (1-ratio) * pic2[x,y,z])
        results.append(new_image)
                    
    return results, labels

if __name__ == '__main__':
    img1 = np.zeros([32,32,1])
    img2 = np.ones([32,32,1])
    interpolated_images, _ = interpolate(100, 10)

    plots = 100
    f, subs = plt.subplots(plots//10, 10, figsize=(16,16))

    for n in range(plots//10):
        for k in range(10):
            subs[n, k].imshow(interpolated_images[10*n+k])            