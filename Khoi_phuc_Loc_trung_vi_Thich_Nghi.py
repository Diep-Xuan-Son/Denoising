import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from math import log10, sqrt

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

def Loc_Trung_Vi_thich_nghi(img,ksize,Smax):     # Định nghĩa hàm lọc trung vị thích nghi
    m,n = img.shape  # lấy 2 chiều của ảnh
    img_ket_qua_anh_loc= np.zeros([m, n]) # Tạo ma trận có kích thước bằng kích thước mxn
                                          # để lưu ảnh kết quả lọc
    h = (Smax-1)//2    # Thêm số pixel vào lề ảnh. Chú ý nếu bộ lọc có kích thước K thì:
                       # số pixel thêm vào mỗi lề ảnh (chú ý: ảnh có 4 lề) là (K-1)/2
    padded_img = np.pad(img,(h,h),mode='reflect')  #Thêm lề ảnh
    for i in range(m):
        for j in range(n):
            k = ksize
            vung_anh_kich_thuoc_k = padded_img[i:i+k,j:j+k] # tạo vùng lân cận (i,j)
                                                                     # Và cũng chính là vùng Sxy
            while True:
                # Bước A
                A1 = np.median(vung_anh_kich_thuoc_k) - np.min(vung_anh_kich_thuoc_k)
                A2 = np.median(vung_anh_kich_thuoc_k) - np.max(vung_anh_kich_thuoc_k)
                if A1 > 0 and A2 <0:
                    # Đi đến Bước B
                    # Chú ý: Giữ liệu các pixel số nguyên trong vùng [0..255]
                    # Nếu không chuyển sang int khi trừ thì chương trình sẽ cảnh báo
                    B1 = int(img[i, j]) - int(np.min(vung_anh_kich_thuoc_k))
                    B2 = int(img[i, j]) - int(np.max(vung_anh_kich_thuoc_k))
                    if B1>0 and B2 <0:
                        img_ket_qua_anh_loc[i,j] = img[i,j]
                    else:
                        img_ket_qua_anh_loc[i, j] = np.median(vung_anh_kich_thuoc_k)
                    break  # Thoát khỏi lặp
                else: # Quay lại bước A
                    k += 1
                    Snew = k*2+1
                    if Snew <= Smax :
                        vung_anh_kich_thuoc_k = padded_img[i:i+k,j:j+k]
                    else :
                        img_ket_qua_anh_loc[i,j] = np.median(vung_anh_kich_thuoc_k)
                        break # Thoát khỏi lặp
    return img_ket_qua_anh_loc

if __name__ == "__main__":
    path = "./Degraded1/"
    lists = os.listdir(path)
    path_save = "./denoising/"

    ksize=7   # Kích thước khởi tạo của bộ lọc thường là số lẽ
    Smax=11    # Kích thước tối đa của bộ lọc

    PSNR_noise_TB = []
    PSNR_denoise1_TB = []
    PSNR_denoise2_TB = []

    SSIM_noise_TB = []
    SSIM_denoise1_TB = []
    SSIM_denoise2_TB = []

    for name in lists:
        img_original = cv2.imread("./original1/" + name,0) # Đọc ảnh
        # img_original = cv2.imread("./original/C 63-line1-cr.png",0) # Đọc ảnh
        img_original = img_original/255

        img_nhieu = cv2.imread("./Degraded1/" + name,0) # Đọc ảnh
        # img_nhieu = cv2.imread("./Degraded/C 63-line1-cr.png",0) # Đọc ảnh
        img_nhieu = img_nhieu/255

        img_ket_qua = Loc_Trung_Vi_thich_nghi(img_nhieu,ksize, Smax)    #Gọi hàm lọc trung vị

        kernelSize = (5,5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
        # erosion = cv2.erode(img_ket_qua,kernel,iterations = 1)
        opening = cv2.morphologyEx(img_ket_qua, cv2.MORPH_OPEN, kernel)

        # img_result = cv2.medianBlur(img_nhieu, 7)

        PSNR_noise = PSNR(img_original, img_nhieu)
        PSNR_denoise1 = PSNR(img_original, img_ket_qua)
        PSNR_denoise2 = PSNR(img_original, opening)
        PSNR_noise_TB.append(PSNR_noise)
        PSNR_denoise1_TB.append(PSNR_denoise1)
        PSNR_denoise2_TB.append(PSNR_denoise2)

        SSIM_noise = ssim(img_original, img_nhieu)
        SSIM_denoise1 = ssim(img_original, img_ket_qua)
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
        # ax2.axis("off")
        # ax2.set_xlabel(f'PSNR noise: {round(PSNR_noise, 3)}\n SSIM noise: {round(SSIM_noise, 3)}', fontsize=12)

        ax3.imshow(img_ket_qua, cmap='gray') # Hiển thị ảnh sau khi lọc
        ax3.set_title("ảnh sau khi qua bộ lọc trung vị thích nghi" + "(" + f'PSNR denoise: {round(PSNR_denoise1, 3)}\n SSIM denoise: {round(SSIM_denoise1, 3)}' + ")") # Thiết lập tiêu đề vùng ax3
        # ax3.axis("off")
        # ax3.set_xlabel(f'PSNR denoise: {round(PSNR_denoise1, 3)}\n SSIM denoise: {round(SSIM_denoise1, 3)}', fontsize=12)

        ax4.imshow(opening, cmap='gray') # Hiển thị ảnh sau khi lọc
        ax4.set_title("ảnh sau khi qua opening" + "(" + f'PSNR denoise: {round(PSNR_denoise2, 3)}\n SSIM denoise: {round(SSIM_denoise2, 3)}' + ")") # Thiết lập tiêu đề vùng ax2
        # ax4.axis("off")
        # ax4.set_xlabel(f'PSNR denoise: {round(PSNR_denoise2, 3)}\n SSIM denoise: {round(SSIM_denoise2, 3)}', fontsize=12)

        plt.savefig(path_save + name + ".png")
        # plt.show()


    print("PSNR_noise_TB: ", sum(PSNR_noise_TB)/len(PSNR_noise_TB))
    print("PSNR_denoise1_TB: ", sum(PSNR_denoise1_TB)/len(PSNR_denoise1_TB))
    print("PSNR_denoise2_TB: ", sum(PSNR_denoise2_TB)/len(PSNR_denoise2_TB))
    print("SSIM_noise_TB: ", sum(SSIM_noise_TB)/len(SSIM_noise_TB))
    print("SSIM_denoise1_TB: ", sum(SSIM_denoise1_TB)/len(SSIM_denoise1_TB))
    print("SSIM_denoise2_TB: ", sum(SSIM_denoise2_TB)/len(SSIM_denoise2_TB))
