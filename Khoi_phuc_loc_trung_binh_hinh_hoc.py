import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from math import log10, sqrt

def anhxam(img0):
    # img = cv2.resize(img0, (500,500))
    img = img0.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(4,4))
    clahe_img = clahe.apply(img)
    # cv2.imshow("Original image", img)
    # cv2.imshow('CLAHE Image', clahe_img)
    return clahe_img

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return np.inf
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel/ sqrt(mse))
    return psnr

def ssim(img1, img2, d_range=1):
    C1 = (0.01 * d_range)**2
    C2 = (0.03 * d_range)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)

    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def Loc_Trung_binh_hinh_hoc(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_TB_cuc_bo = np.mean(vung_anh_kich_thuoc_k)
            gia_tri_loc = np.prod(vung_anh_kich_thuoc_k) ** (1.0 / m * n)
            if gia_tri_loc > gia_tri_TB_cuc_bo:
               img_ket_qua_anh_loc[i, j]= int(gia_tri_TB_cuc_bo)
            else:
               img_ket_qua_anh_loc[i,j] = int(gia_tri_loc)
    return img_ket_qua_anh_loc

if __name__ == "__main__":
    path = "./Degraded1/"
    lists = os.listdir(path)
    path_save = "./denoising/"

    ksize =3
    kernelSize = (5,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)

    PSNR_noise_TB = []
    PSNR_denoise1_TB = []
    PSNR_denoise2_TB = []

    SSIM_noise_TB = []
    SSIM_denoise1_TB = []
    SSIM_denoise2_TB = []

    for name in lists:
        img_original = cv2.imread("./original1/" + name,0) # Đọc ảnh
        img_original = img_original/255

        img_nhieu = cv2.imread("./Degraded1/" + name,0) # Đọc ảnh
        img_nhieu = img_nhieu/255
        img_ket_qua_TBHH = Loc_Trung_binh_hinh_hoc(img_nhieu, ksize)

        opening = cv2.morphologyEx(img_ket_qua_TBHH, cv2.MORPH_OPEN, kernel)

        PSNR_noise = PSNR(img_original, img_nhieu)
        PSNR_denoise1 = PSNR(img_original, img_ket_qua_TBHH)
        PSNR_denoise2 = PSNR(img_original, opening)
        PSNR_noise_TB.append(PSNR_noise)
        PSNR_denoise1_TB.append(PSNR_denoise1)
        PSNR_denoise2_TB.append(PSNR_denoise2)

        SSIM_noise = ssim(img_original, img_nhieu)
        SSIM_denoise1 = ssim(img_original, img_ket_qua_TBHH)
        SSIM_denoise2 = ssim(img_original, opening)
        SSIM_noise_TB.append(SSIM_noise)
        SSIM_denoise1_TB.append(SSIM_denoise1)
        SSIM_denoise2_TB.append(SSIM_denoise2)

        fig = plt.figure(figsize=(16, 9))     # Thiết lập vùng (cửa sổ) vẽ
        [(ax1, ax2), (ax3,ax4)] = fig.subplots(2, 2)
        ax1.imshow(img_original, cmap='gray')      # Hiển thị ảnh gốc vùng ax1
        ax1.set_title("ảnh ban đầu")    # Thiết lập tiêu đề vùng ax1
        ax1.axis("off")

        ax2.imshow(img_nhieu, cmap='gray')      # Hiển thị ảnh gốc vùng ax2
        ax2.set_title("ảnh nhiễu" + "(" + f'PSNR noise: {round(PSNR_noise, 3)}\n SSIM noise: {round(SSIM_noise, 3)}' + ")")    # Thiết lập tiêu đề vùng ax2

        ax3.imshow(img_ket_qua_TBHH, cmap='gray')  # Hiển thị ảnh sau khi lọc
        ax3.set_title("ảnh sau khi lọc trung bình hình học" + "(" + f'PSNR denoise: {round(PSNR_denoise1, 3)}\n SSIM denoise: {round(SSIM_denoise1, 3)}' + ")")  # Thiết lập tiêu đề vùng ax2

        ax4.imshow(opening, cmap='gray') # Hiển thị ảnh sau khi lọc
        ax4.set_title("ảnh sau khi qua opening" + "(" + f'PSNR denoise: {round(PSNR_denoise2, 3)}\n SSIM denoise: {round(SSIM_denoise2, 3)}' + ")")

        # plt.show()
        plt.savefig(path_save + name + ".png")

    print("PSNR_noise_TB: ", sum(PSNR_noise_TB)/len(PSNR_noise_TB))
    print("PSNR_denoise1_TB: ", sum(PSNR_denoise1_TB)/len(PSNR_denoise1_TB))
    print("PSNR_denoise2_TB: ", sum(PSNR_denoise2_TB)/len(PSNR_denoise2_TB))
    print("SSIM_noise_TB: ", sum(SSIM_noise_TB)/len(SSIM_noise_TB))
    print("SSIM_denoise1_TB: ", sum(SSIM_denoise1_TB)/len(SSIM_denoise1_TB))
    print("SSIM_denoise2_TB: ", sum(SSIM_denoise2_TB)/len(SSIM_denoise2_TB))