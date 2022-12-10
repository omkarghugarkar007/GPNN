#%%
import os
import numpy as np
# os.chdir('/root/evaluate-saliency-5/GPNN')
#%%
# faiss doesnt work without gpu
# !python random_sample.py -in database/balloons.png --faiss
import skimage.io
from matplotlib import pyplot as plt
# original_imname = 'images/ILSVRC2012_val_00000013.JPEG'
# original_imname = 'database/balloons.png'
original_imname = 'database/volacano.png'
output_imname = os.path.join('output',os.path.basename(original_imname))
output_imname_root,ext = output_imname.split('.')
output_imname = output_imname_root + '_random_sample' +'.png'
original_im = skimage.io.imread(original_imname)
# !python random_sample.py -in  {original_imname} --faiss
# assert False
from model.my_gpnn import gpnn
# from model.gpnn import gpnn
config = {
    'out_dir':'.',
    'iters':10,
    'coarse_dim':14,
    'out_size':0,
    'patch_size':7,
    'stride':1,
    'pyramid_ratio':4/3,
    'faiss':True,
    'no_cuda':False,
    #---------------------------------------------
    'in':None,
    'sigma':0.75,
    'alpha':0.005,
    'task':'random_sample',
    #---------------------------------------------
#     'input_img':original_im,
    'input_img':original_imname,

}

model = gpnn(config)
output_im = model.run(to_save=True)
    

# output_im = skimage.io.imread(output_imname)

if False:
    plt.figure()
    plt.imshow(np.array(original_im[...,:3]))
    plt.show()

    plt.figure()
    plt.imshow(np.array(output_im))
    plt.show()
#%%
