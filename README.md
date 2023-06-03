# cycleGAN
Cycle GAN implementation here in this Github page

About Code:
- model.py has generator and discriminator structre
- train.py has configirations for training which can be changed and also implementation of model.py classes
- test.py has test configirations for testing models from training (by default 10th batch but can be easily modified)

About Classes and Images:

a is Thermal Image class which is black and white and b is rgb colored images from same dataset but they are not paired while I selected 1/3rd of the dataset so not all thermal images have rgb pair or vice versa. Class structre is made like this because dataloader on train and test scripts want single folder inside target folder that has only one classes images
