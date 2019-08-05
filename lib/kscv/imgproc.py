import cv2
import numpy as np
import numpy.random as rd
import skimage

####################### color space ######################
def bgr2hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return h, s, v

def bgr2yuv(rgb):
    yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV_I420)
    yuv = yuv.flatten()
    h, w = rgb.shape[:2]

    Y = yuv[:w*h].reshape(h, w)
    U = yuv[w*h:w*h*5/4].reshape(h/2, w/2)
    V = yuv[w*h*5/4:].reshape(h/2, w/2)

    return Y, U, V


####################### transform ########################
def random_crop(img, 
                w=1, 
                h=1, 
                aspect_ratio=1.0, 
                fix_aspect_ratio=False,
                min_ratio=0.1,
                max_ratio=0.9):
    img_h, img_w = img.shape[:2]
    assert 0<w<=img_w and 0<h<=img_h and max_ratio>=min_ratio
    img_aspect_ratio = img_w*1.0/img_h

    if fix_aspect_ratio:
        h = int(w*1.0/aspect_ratio)
        assert 0<h<=img_h
        if aspect_ratio>=img_aspect_ratio:
            min_h = int(min_ratio*img_w/aspect_ratio)
            max_h = int(max_ratio*img_w/aspect_ratio)
        else:
            min_h = int(min_ratio*img_h)
            max_h = int(max_ratio*img_h)
            

        crop_h = rd.randint(min_h, max_h+1)
        crop_w = int(aspect_ratio*crop_h)
        crop_l = rd.randint(img_w-crop_w+1)
        crop_t = rd.randint(img_h-crop_h+1)
        crop_r = crop_l + crop_w
        crop_b = crop_t + crop_h
        patch = img[crop_t:crop_b, crop_l:crop_r, :]
    else:
        crop_w = w
        crop_h = h
        crop_l = rd.randint(img_w-crop_w)
        crop_t = rd.randint(img_h-crop_h)
        crop_r = crop_l + crop_w
        crop_b = crop_t + crop_h

        patch = img[crop_t:crop_b, crop_l:crop_r, :]

    return patch


#################### sharpen image ###################
def gaussian_sharpen(img, factor=0.5):
    blur_img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, factor+1, blur_img, -factor, 0)
    return img

def kernel_sharpen(img, factor=0.2):
    kernel = np.zeros([3, 3]).astype(np.float) - factor
    kernel[1, 1] = kernel[1, 1] + factor*9 + 1
    img = cv2.filter2D(img, -1, kernel)
    return img

##################### image noise ####################
JPEG_FLAG = int(cv2.IMWRITE_JPEG_QUALITY)
def add_jpeg_noise(img, quality=20):
    jpeg_data = cv2.imencode(
                    '.jpg',
                    img,
                    [JPEG_FLAG, quality])[1]
    restore_img = cv2.imdecode(
                    np.array(jpeg_data),
                    cv2.IMREAD_UNCHANGED)
    return restore_img

def add_gaussian_blur(img, ksize=3):
    blur_img = cv2.GaussianBlur(
                    img,
                    (ksize, ksize),
                    1)
    return blur_img

def add_upscale_noise(img, scale_factor=2, iter_num=1):
    img_h, img_w = img.shape[:2]

    scale_h = int(img_h/scale_factor)
    scale_w = int(img_w/scale_factor)

    for i in range(iter_num):
        scale_img = cv2.resize(img, 
                        (scale_w, scale_h),
                        0,
                        0,
                        cv2.INTER_NEAREST)
        img = cv2.resize(scale_img,
                        (img_w, img_h),
                        0,
                        0,
                        cv2.INTER_NEAREST)
    return img

def add_gaussian_noise(img, var):
    noise_img = skimage.util.random_noise(
                        img,
                        mode='gaussian',
                        mean=0,
                        var=var,
                        seed=None,
                        clip=True
                        )
    noise_img = (noise_img*255).astype(np.uint8)
    return noise_img

####################### Image Quality #####################
def calc_sharpness(img_gray):
    gy, gx = np.gradient(img_gray)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness

def calc_blurrness(img):
    blurrness = cv2.Laplacian(img, cv2.CV_64F).var()
    return blurrness


