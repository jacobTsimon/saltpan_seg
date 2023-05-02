import numpy as np
from PIL import Image
import os
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import rasterio as rio

# img35 = rio.open('./train_scenes/20220114_151635_96_245c_3B_AnalyticMS_SR.tif')
# plt.imshow(img35.read(2),cmap = "Accent")
# plt.show()

dir = './train_scenes/'

for root, _,file in os.walk(dir):
    for names in file:
        path = root + names
        print(path)
        print(type(file))

        img = rio.open('./train_scenes/20220114_151635_96_245c_3B_AnalyticMS_SR.tif') #Image.open(path)
        array = img.read()
        array = array.astype('float64')
        print(array.shape)
        #img.show()
        tens = ToTensor()(array)
        print(tens)
        c1 = tens[:,0,:]
        # plt.imshow(c1,cmap = "Accent")
        # plt.show()

        c2 = tens[:,1,:]
        c3 = tens[:,2,:]
        c4 = tens[:,3,:]

        print(tens.size())
        print(np.unique(c1))
        print(c1.max(),c1.min())
        print(np.unique(c2))
        print(c2.max(), c2.min())
        print(np.unique(c3))
        print(c3.max(), c3.min())
        print(np.unique(c4))
        print(c4.max(), c4.min())
        # n1 = c1[~torch.any(tens.isnan(), dim=0)]
        # u = np.unique(n1)

        # print(np.unique(u))
        # print(n1.max(),n1.min())
        #print(tens[3, :, :].min())

        # print(torch.any(tens.isnan(),dim= 1))
        # tf = torch.any(tens.isnan(),dim= 0)
        # print(tf.shape)
        # tf_int = tf.long()
        # print(tf_int.shape)
        #
        # plt.imshow(tf_int, cmap="Accent")
        # plt.show()




