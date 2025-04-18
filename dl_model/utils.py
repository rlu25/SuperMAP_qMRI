import torch
import numpy as np



"""
Helper functions for new types of inverse problems
"""




def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  x = torch.fft.ifftshift(x, dim=[-1, -2])
  return torch.fft.fftshift(torch.fft.fft2(x, dim=[-1, -2],norm='ortho'),dim=[-1,-2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]),dim=[-1,-2],norm='ortho')
  return  torch.fft.fftshift(x,dim=[-1,-2])

def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= torch.min(img)
  img /= torch.max(img)
  return img


def normalize_np(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def normalize_complex(img):
  """ normalizes the magnitude of complex-valued image to range [0, 1] """
  abs_img = normalize(torch.abs(img))
  ang_img = normalize(torch.angle(img))
  return abs_img * torch.exp(1j * ang_img)

def kspace_to_image(kdata):
    b,c,h,w = kdata.shape
    k_real = kdata[:,0,:,:]
    k_real = k_real.reshape(b,1,h,w)
    k_imag = kdata[:,1,:,:]
    k_imag = k_imag.reshape(b,1,h,w)
    k_complex = k_real + 1j * k_imag
    img_complex = ifft2(k_complex)
    img_real = torch.real(img_complex)
    img_imag = torch.imag(img_complex)
    img = torch.cat((img_real, img_imag), dim=1)
    return img


def image_to_kspace(img):
    b,c,h,w = img.shape
    img_real = img[:,0,:,:]
    img_real=img_real.reshape(b,1,h,w)
    img_imag=img[:,1,:,:]
    img_imag=img_imag.reshape(b,1,h,w)
    img_complex = img_real + 1j * img_imag
    k_complex = fft2(img_complex)
    k_real = torch.real(k_complex)
    k_imag = torch.imag(k_complex)
    kdata = torch.cat((k_real, k_imag), dim=1)
    return kdata

