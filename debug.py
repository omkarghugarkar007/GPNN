import gradcam
import numpy as np 
import torch 

input = torch.zeros(100,3,168,250)
cams = gradcam.gradcam(input,target=None)