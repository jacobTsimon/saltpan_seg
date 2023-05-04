import torch
import torchvision
import torchgeo
import rasterio as rio
import torchvision.transforms as transforms
import numpy as np
from torchgeo.transforms import indices
import matplotlib.pyplot as plt
from PIL import Image



##APPEND NDVI & ELEVATION and convert back?

##code for appending NDVI


##RATIO OF BACKGROUND TO PAN PIXELS IS 1280:1

##code for saving tensor as a new tiff
# import tifffile
#
# tensor35 = tensor35[0]
# tifffile.imsave(savefile,tensor35)
#
# test = tifffile.imread(savefile)
#
# np.all(np.equal(tensor35,test))
##try using torchvision.utils.save_image ... FAILED
# ndvi_ind = torch.tensor([5])

# ndvi = torch.index_select(tensor35,0,ndvi_ind)
# print(ndvi)
# testB = ndvi == tensor35[ndvi_ind]
# print(torch.all(testB == True))
# ndvi = (ndvi + 1)/2
# print(ndvi)
# testA = ndvi == tensor35[ndvi_ind]
# print(torch.all(testA == True))
# tensor35[ndvi_ind] = ndvi
# testF = ndvi == tensor35[ndvi_ind]
# print(torch.all(testF == True))
#
#
# torchvision.utils.save_image(tensor35,savefile)

##code for visualizing NDVI
#
# plt.imshow(tensor35[-1],cmap = "RdYlGn_r")
# plt.show()

##CREATE RANDOM SAMPLER

#create RandomGeoSampler
from torchgeo.datasets import RasterDataset, unbind_samples,stack_samples
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader
from torchgeo.samplers import Units

#identify folder containing data
trainDS = './train_scenes'
trainTRUTH = './train_truth'

#create raster dataset (from custom raster doc: https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html)
class PlanetScope(RasterDataset):

    #transforms = transform
    filename_glob = "2022*_3B_*.tif"
    filename_regex = "^(?P<date>\d{8}_\d{6})_.{2}_.{4}_(3B_*).*"
    date_format = "%Y%m%d_%H%M%S"
    is_image = True
    separate_files = False
    all_bands = ["1", "2", "3", "4","5"]
    rgb_bands = ["1", "2", "3"]

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image

        image = sample["image"]
        print(image.shape)

        image = image[rgb_indices].permute(1, 2, 0)
        print(image.shape)
        print(type(image))

        #image = torch.clamp(image , min=0, max=1).numpy()
        image = image.numpy().astype(int)


        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig

class ElevationData(RasterDataset):

    #transforms = transform
    filename_glob = "elevationLayer*.tif"
    filename_regex = "^elevation.*"

    is_image = True
    separate_files = False
    all_bands = ["1"]

##visualization taken from:
from typing import Iterable, List

def plot_imgs(images: Iterable, axs: Iterable, chnls: List[int] = [2, 1, 0], bright: float = 3.):
    for img, ax in zip(images, axs):
        #arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = img.permute(1, 2, 0).numpy().astype(int) #[:, :, chnls]
        ax.imshow(rgb)
        ax.axis('off')


def plot_msks(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy(), cmap='Blues')
        ax.axis('off')


def plot_batch(batch: dict, bright: float = 3., cols: int = 4, width: int = 5, chnls: List[int] = [2, 1, 0]):
    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())

    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ('image' in batch) and ('mask' in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ('image' in batch) and ('mask' in batch):
        # plot the images on the even axis
        plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1)[::2], chnls=chnls,
                  bright=bright)  # type: ignore

        # plot the masks on the odd axis
        plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1)[1::2])  # type: ignore

    else:

        if 'image' in batch:
            plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1), chnls=chnls,
                      bright=bright)  # type: ignore

        elif 'mask' in batch:
            plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1))  # type: ignore


dataset = PlanetScope(trainDS)

class PlanetMask(RasterDataset):
    filename_glob = "pan*F.tif"
    filename_regex = "^pan.*"
    date_format = None
    is_image = False
    separate_files = False
    all_bands = ["1"]
# #create mask dataset
# truth = PlanetMask(trainTRUTH)
#
# #combine the data and mask sets
# train_dataset = dataset & truth
# print(train_dataset)
#
# #PUT ROI INTO sampler
# mint, maxt = 1642191395.0, 1642191395.999999 #need to figure this for time series !!!
# #roi needs to be in ... pixels?
# #roi here is based on 35half from GEOAI_proj
# roi = torchgeo.datasets.BoundingBox(minx = 447139.7152,
#                                     maxx =509083.5446,
#                                     miny = 3464718.6179,
#                                     maxy = 3499982.3511,
#                                     mint = mint,
#                                     maxt = maxt)
# sampler = RandomGeoSampler(train_dataset, size=512 * 3, length=10,units=Units.CRS,roi = roi) #
# dataloader = DataLoader(train_dataset, sampler=sampler, collate_fn=stack_samples) #
#
# for batch in dataloader:
#     sample = unbind_samples(batch)[0]
#     print(sample["image"].shape)
#     print(type(sample))
#     #we can do the NDVI transform here
#     imN = transform(sample["image"])[0]
#     print(imN.shape)
#     #could this be a problem? since we are doing the transform after random sampling?
#
#     plot_batch(batch)
#
#     #dataset.plot(sample)
#     plt.axis("off")
#     plt.show()