from pylab import *
import os
import cv2
import torch
from Dataset import dataset
#from albumentations.pytorch.functional import img_to_tensor
from U_Net.unet_model import UNet
#from Analysis_experiment.MSRAU_UNet_V3_DS.unet_model import UNet


def get_model(model_path):
     
    model = UNet(1,1)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model

def mask_overlay(image, mask, color=255):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack(mask * np.array(color))
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.1, image, 0.9, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img



model_path = './U-Net_54.pt'
model = get_model(model_path)

img_list = os.listdir('./data/new_val/images')
for i in img_list:
    if i == '.ipynb_checkpoints':
        continue
    img_file_name = os.path.join('./data/new_val/images',i)
    image = dataset.load_img(img_file_name)
    with torch.no_grad():
        input_image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image).cuda(), dim=0),dim=0)
    mask = model(input_image)
    mask_array = mask.data[0].cpu().numpy()[0]
    mask_array[mask_array>0.5] = 255
    mask_array[mask_array != 255] = 0
    #M = cv2.getRotationMatrix2D((256,256), 90, 1.0)
    #rotated = cv2.warpAffine(mask_array, M, (512, 512))
    #img_end =cv2.flip(rotated,0) 
    cv2.imwrite(os.path.join("./demo_result/UNet",i),mask_array)
#     cv2.imshow('result',mask_array)
#     cv2.waitKey(0) #等待时间，单位毫秒，0为任意键终止
#     cv2.destroyAllWindows() #销毁所有窗口
    print("waiting!")
print("Successfully!")



'''
origin_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
img_end = cv2.cvtColor(img_end, cv2.COLOR_GRAY2BGR)



for h in range(img_end.shape[0]):
    for w in range(img_end.shape[1]):
            if(img_end[h,w,0] == 255 and img_end[h,w,1] == 255 and img_end[h,w,2] == 255 ):
                img_end[h,w,0]= 227
                img_end[h,w,1]= 247
                img_end[h,w,2]= 6

cv2.imwrite('./mask/769_10_0.bmp',img_end)
dst=cv2.imread('./mask/769_10_0.bmp')



weighted_sum = cv2.addWeighted(dst, 0.3, origin_img,0.7,0)              



#cv2.waitKey(0)
cv2.imwrite('./769_10.bmp',weighted_sum)
'''