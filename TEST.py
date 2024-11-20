"""
TODO
python your_script_name.py path_to_weight_file path_to_input_directory path_to_output_directory
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision import transforms as T
from torchvision import datasets
import  torchvision
import cv2
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct, idct
from model import Stegnet

from model import ResidualBlockNoBN2
from model import  ResidualDenseBlock_out_drop

import pytorch_msssim
import math
from tiramisu import FCDenseNet

import cv2

from resnet_cbam import CBAM_2MLP



def dwt_init(x, device):
    x = x.to(device)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    h = torch.zeros((x_LL.size(0), x_LL.size(1), x_LL.size(2) * 2, x_LL.size(3) * 2),device=device)

    h[:, :, 0::2, 0::2] = x_LL
    h[:, :, 1::2, 0::2] = x_HL
    h[:, :, 0::2, 1::2] = x_LH
    h[:, :, 1::2, 1::2] = x_HH


    return h

def iwt_init(x, device):
    x = x.to(device)

    x_LL = x[:, :, 0::2, 0::2]
    x_HL = x[:, :, 1::2, 0::2]
    x_LH = x[:, :, 0::2, 1::2]
    x_HH = x[:, :, 1::2, 1::2]

    x1 = x_LL - x_HL - x_LH + x_HH
    x2 = x_LL - x_HL + x_LH - x_HH
    x3 = x_LL + x_HL - x_LH - x_HH
    x4 = x_LL + x_HL + x_LH + x_HH

    h = torch.zeros((x1.size(0), x1.size(1), x1.size(2) * 2, x1.size(3) * 2),device=device)

    h[:, :, 0::2, 0::2] = x1
    h[:, :, 1::2, 0::2] = x2
    h[:, :, 0::2, 1::2] = x3
    h[:, :, 1::2, 1::2] = x4

    return h

def round_diff(x):
    sign = torch.ones_like(x)
    sign[torch.floor(x) % 2 == 0] = -1
    y = sign * torch.cos(x * torch.pi) / 2
    out = torch.round(x) + y - y.detach()
    return out

BATCH_SIZE =1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


WEIGHT_PATH = 'D:\\code_yinxie\\Stegnet-Covertcast-main_torch\\Stegnet-Covertcast-main\\stegnet\\a_CBAM2MLP_ONLY\\2024-02-14_12-01-36\\weights\\epoch-100.pt'
WEIGHT_PATH1 ='D:\\code_yinxie\\Stegnet-Covertcast-main_torch\\Stegnet-Covertcast-main\\stegnet\\a_CBAM2MLP_2enhance_2step_SCCOVE_2dense_dropout\\2024-05-16_14-14-40\\weights\\epoch-100_enhance.pt'
test_cover_PATH ='D:\\code_yinxie\\811data\\test'
test_secret_PATH ='D:\\code_yinxie\\811data\\test'
OUTPUT_PATH ='D:\\code_yinxie\\Stegnet-Covertcast-main_torch\\Stegnet-Covertcast-main\\stegnet\\output811\\a_ 2mpl_final20241111\\tset'

data_transform = T.Compose([
	T.Resize((256,256)),
	T.ToTensor(),

])


test_cover_dataset = datasets.ImageFolder(os.path.join(test_cover_PATH, 'cover'), transform=data_transform)
test_secret_dataset =datasets.ImageFolder(os.path.join(test_secret_PATH, 'secret'), transform=data_transform)

test_cover_dataloader = DataLoader(test_cover_dataset, batch_size=BATCH_SIZE,
								shuffle=False, drop_last=False)
test_secret_dataloader = DataLoader(test_secret_dataset, batch_size=BATCH_SIZE,
								shuffle=False, drop_last=False)

precoder=CBAM_2MLP().to(DEVICE)

resnet = ResidualBlockNoBN2().to(DEVICE)
postEnhance = ResidualDenseBlock_out_drop().to(DEVICE)
encoder = FCDenseNet().to(DEVICE)


decoder = Stegnet(3).to(DEVICE)

checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')

encoder.load_state_dict(checkpoint['encoder_dict'])
decoder.load_state_dict(checkpoint['decoder_dict'])
precoder.load_state_dict(checkpoint['precoder_dict'])

checkpoint1 = torch.load(WEIGHT_PATH1, map_location='cpu')
resnet.load_state_dict(checkpoint1['resnet_dict'])
postEnhance.load_state_dict(checkpoint1['postEnhance_dict'])

cover_mse = 0.0
secret_mse = 0.0
quantized_secret_mse = 0.0
quantized_outputstensor_secret_mse = 0.0

ssim_emb_cov = 0.0
ssim_out_secret = 0.0
ssim_quati_secr = 0.0

psnr_emb_cov = 0.0
psnr_out_secret = 0.0
psnr_quai_secret =0.0
output_folder_de_secret = os.path.join(OUTPUT_PATH, 'de_secret')
os.makedirs(output_folder_de_secret, exist_ok=True)
output_folder_cover = os.path.join(OUTPUT_PATH, 'cover')
os.makedirs(output_folder_cover, exist_ok=True)
output_folder_secretpath = os.path.join(OUTPUT_PATH, 'secret')
os.makedirs(output_folder_secretpath, exist_ok=True)
output_folder_quai_secretpath = os.path.join(OUTPUT_PATH, 'quai_secret')
os.makedirs(output_folder_quai_secretpath, exist_ok=True)
output_folder_nosuofang_embed = os.path.join(OUTPUT_PATH, 'nosuo_embed')
os.makedirs(output_folder_nosuofang_embed, exist_ok=True)
output_folder_nosuofang_decode = os.path.join(OUTPUT_PATH, 'nosuo_decode')
os.makedirs(output_folder_nosuofang_decode, exist_ok=True)

output_folder_emb_pad = os.path.join(OUTPUT_PATH, 'emb_pad')
os.makedirs(output_folder_emb_pad, exist_ok=True)
output_folder_iwt_pad = os.path.join(OUTPUT_PATH, 'iwt_pad')
os.makedirs(output_folder_iwt_pad, exist_ok=True)
output_folder_iwt = os.path.join(OUTPUT_PATH, 'iwt')
os.makedirs(output_folder_iwt, exist_ok=True)
output_folder_cbam_cover = os.path.join(OUTPUT_PATH, 'cbam_cover')
os.makedirs(output_folder_cbam_cover, exist_ok=True)
output_folder_cbam_secret = os.path.join(OUTPUT_PATH, 'cbam_secret')
os.makedirs(output_folder_cbam_secret, exist_ok=True)

output_folder_emb_cove_20 = os.path.join(OUTPUT_PATH, 'emb_cover20')
os.makedirs(output_folder_emb_cove_20, exist_ok=True)
output_folder_desecert_secret_20 = os.path.join(OUTPUT_PATH, 'descere_secret20')
os.makedirs(output_folder_desecert_secret_20, exist_ok=True)
output_folder_cbam_cover_cover = os.path.join(OUTPUT_PATH, 'cbam_cover_scover20t')
os.makedirs(output_folder_cbam_cover_cover, exist_ok=True)
output_folder_cbam_secret_secret = os.path.join(OUTPUT_PATH, 'cbam_secret_secret20')
os.makedirs(output_folder_cbam_secret_secret, exist_ok=True)

output_folder_emb_2 = os.path.join(OUTPUT_PATH, 'emb_2')
os.makedirs(output_folder_emb_2, exist_ok=True)

output_folder_emb_coverpre = os.path.join(OUTPUT_PATH, 'emb_coverpre')
os.makedirs(output_folder_emb_coverpre, exist_ok=True)

emb_cove_98 = os.path.join(OUTPUT_PATH, 'emb_cover98')
os.makedirs(emb_cove_98, exist_ok=True)

emb_cove_98_20 = os.path.join(OUTPUT_PATH, 'emb_cover98_20')
os.makedirs(emb_cove_98_20, exist_ok=True)
def computePSNR(origin,pred):
    mse =F.mse_loss(origin,pred).item()
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

with torch.no_grad():

	for i, (test_cover_data,test_secret_data) in tqdm(enumerate(zip(test_cover_dataloader,test_secret_dataloader), start=1)):

		test_cover_input, _ =test_cover_data
		test_secret_input, _ = test_secret_data

		test_covers = test_cover_input.to(DEVICE)
		test_secrets = test_secret_input.to(DEVICE)
		test_cover_cbam = precoder(test_covers)
		test_secrets_cbam = precoder(test_secrets)

		test_secret_dct = dwt_init(test_secrets_cbam, DEVICE)
		top_padding_size =8
		test_covers_1 = F.pad(test_cover_cbam,
							  (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
							  mode='reflect')
		test_secrets_1 = F.pad(test_secret_dct,
							   (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
							   mode='reflect')

		original_height, original_width = test_secrets_1.shape[2], test_secrets_1.shape[3]
		embeds = encoder(torch.cat([test_covers_1,test_secrets_1], dim=1))
		top_crop = top_padding_size
		bottom_crop = original_height - top_padding_size
		left_crop = top_padding_size
		right_crop = original_width - top_padding_size
		embeds = embeds[:, :, top_crop:bottom_crop, left_crop:right_crop]

		test_secret_dct_2 = dwt_init(test_secrets, DEVICE)
		test_covers_2 = F.pad(test_covers,
							  (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
							  mode='reflect')
		test_secrets_2 = F.pad(test_secret_dct_2,
							   (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
							   mode='reflect')

		original_height, original_width = test_secrets_2.shape[2], test_secrets_2.shape[3]
		embeds_2 = encoder(torch.cat([test_covers_2, test_secrets_2], dim=1))
		top_crop = top_padding_size
		bottom_crop = original_height - top_padding_size
		left_crop = top_padding_size
		right_crop = original_width - top_padding_size
		embeds_2 = embeds_2[:, :, top_crop:bottom_crop, left_crop:right_crop]

		embeds_pad = F.pad(embeds, (8, 8, 8, 8), mode='reflect')
		outputs_dct = decoder(embeds)

		original_height2, original_width2 = embeds_pad.shape[2], embeds_pad.shape[3]

		top_crop2 = top_padding_size
		bottom_crop2 = original_height2 - top_padding_size
		left_crop2 = top_padding_size
		right_crop2 = original_width2 - top_padding_size

		outputs = iwt_init(outputs_dct, DEVICE)

		outputs_pad11=outputs

		ContainerImgName = '%s/diff_emb-  cover_epoch%03d_batch%04d.png' % (emb_cove_98, 1, i)
		vutils.save_image(embeds-test_covers, ContainerImgName, nrow=1, padding=1, normalize=True)

		ContainerImgName20 = '%s/diff_emb-cover_epoch%03d_batch%04d.png' % (emb_cove_98_20, 1, i)
		vutils.save_image((embeds - test_covers)*20, ContainerImgName20, nrow=1, padding=1, normalize=True)

		cover_mse += F.mse_loss(embeds, test_covers).item()
		secret_mse += F.mse_loss(outputs,test_secrets).item()

		ssim_emb_cov += ( pytorch_msssim.ssim(embeds, test_covers, data_range=embeds.max() - embeds.min()))
		ssim_out_secret +=( pytorch_msssim.ssim(outputs,test_secrets, data_range=outputs.max() - outputs.min()))

		psnr_emb_cov += computePSNR(embeds, test_covers)
		psnr_out_secret +=   computePSNR(outputs,test_secrets)

		np_embeds = np.transpose(embeds.detach().cpu().numpy(), (0, 2, 3, 1))
		np_embeds = np_embeds * 255


		np_embeds_nosuofang = np_embeds
		quantized_embeds_nosuofang = np.clip(np_embeds_nosuofang, 0, 255).astype(np.uint8)
		min_np_embedsvalue = int(np.min(np_embeds))-1
		max_np_embedsvalue =int( np.max(np_embeds))+1


		np_embeds = (np_embeds - min_np_embedsvalue) / (max_np_embedsvalue -min_np_embedsvalue) * 255
		quantized_embeds = np.clip(np_embeds, 0, 255).astype(np.uint8)

		quantized_tensor =(quantized_embeds.astype(np.float32) / 255) * (max_np_embedsvalue - min_np_embedsvalue) + min_np_embedsvalue
		min_quantized_tensorsvalue =int( np.min(quantized_tensor))
		max_quantized_tensorvalue = int(np.max(quantized_tensor))
		quantized_tensor = quantized_tensor / 255
		quantized_tensor = torch.as_tensor(np.transpose(quantized_tensor, (0, 3, 1, 2)),
											device=DEVICE)



		quantized_tensor_nosuofang =quantized_embeds_nosuofang.astype(np.float32) / 255

		quantized_tensor_nosuofang = torch.as_tensor(np.transpose(quantized_tensor_nosuofang, (0, 3, 1, 2)),
										   device=DEVICE)

		quantized_nosuofang_outputs = decoder(quantized_tensor_nosuofang)
		quantized_nosuofang_outputs = iwt_init(quantized_nosuofang_outputs, DEVICE)
		np_quantized_nosuofang_outputs = np.transpose(quantized_nosuofang_outputs.detach().cpu().numpy(), (0, 2, 3, 1))

		np_quantized_nosuofang_outputs = np_quantized_nosuofang_outputs * 255
		np_quantized_nosuofang_outputs22 = np.clip(np_quantized_nosuofang_outputs, 0, 255).astype(np.uint8)
		quantized_tensor1=resnet(quantized_tensor)
		quantized_outputs = decoder(quantized_tensor1)
		quantized_outputs =postEnhance(quantized_outputs)
		quantized_outputs = iwt_init(quantized_outputs, DEVICE)

		quantized_secret_mse += F.mse_loss(quantized_outputs,test_secrets).item()
		ssim_quati_secr += ( pytorch_msssim.ssim(quantized_outputs, test_secrets, data_range=quantized_outputs.max() - quantized_outputs.min()))

		psnr_quai_secret += computePSNR(quantized_outputs, test_secrets)
		np_outputs = np.transpose(outputs.detach().cpu().numpy(), (0, 2, 3, 1))

		np_outputs = np_outputs * 255
		quantized_outputs_num = np.clip(np_outputs, 0, 255).astype(np.uint8)


		np_cover = np.transpose(test_covers.detach().cpu().numpy(), (0, 2, 3, 1))

		np_cover = np_cover * 255
		quantized_cover_num_16 = np.clip(np_cover, 0, 255).astype(np.int16)
		quantized_cover_num = np.clip(np_cover, 0, 255).astype(np.uint8)



		np_secret = np.transpose(test_secrets.detach().cpu().numpy(), (0, 2, 3, 1))

		np_secret = np_secret * 255
		quantized_secret_num_16 = np.clip(np_secret, 0, 255).astype(np.int16)
		quantized_secret_num = np.clip(np_secret, 0, 255).astype(np.uint8)
		np_qua_de_secret = np.transpose(quantized_outputs.detach().cpu().numpy(), (0, 2, 3, 1))

		np_qua_de_secret =np_qua_de_secret * 255
		quantized_de_secret_num = np.clip(np_qua_de_secret, 0, 255)
		quantized_de_secret_num = np.round(quantized_de_secret_num).astype(np.uint8)
		np_qua_secret_pad = np.transpose(embeds_pad.detach().cpu().numpy(), (0, 2, 3, 1))
		np_qua_secret_pad = np_qua_secret_pad * 255
		np_qua_secret_pad_num = np.clip(np_qua_secret_pad, 0, 255).astype(np.uint8)

		np_qua_secret_outputs_pad11 = np.transpose(outputs_pad11.detach().cpu().numpy(), (0, 2, 3, 1))
		np_qua_secret_outputs_pad11 = np_qua_secret_outputs_pad11 * 255
		np_qua_secret_pad_outputs_pad11 = np.clip(np_qua_secret_outputs_pad11, 0, 255).astype(np.uint8)

		np_qua_secret_outputs_pad = np.transpose(outputs.detach().cpu().numpy(), (0, 2, 3, 1))

		np_qua_secret_outputs_pad = np_qua_secret_outputs_pad * 255
		np_qua_secret_pad_outputs_pad = np.clip(np_qua_secret_outputs_pad, 0, 255).astype(np.uint8)


		np_cover_cbam = np.transpose(test_cover_cbam.detach().cpu().numpy(), (0, 2, 3, 1))

		np_cover_cbam = np_cover_cbam * 255
		quantized_np_cover_cbam_num = np.clip(np_cover_cbam, 0, 255).astype(np.uint8)

		np_cover_cbam_dwt = np.transpose(test_cover_cbam.detach().cpu().numpy(), (0, 2, 3, 1))

		np_cover_cbam_dwt = np_cover_cbam_dwt * 255
		quantized_np_cover_cbam_dwt_num = np.clip(np_cover_cbam_dwt, 0, 255).astype(np.uint8)



		np_secret_ddwt_cbam = np.transpose(test_secret_dct.detach().cpu().numpy(), (0, 2, 3, 1))

		np_secret_ddwt_cbam = np_secret_ddwt_cbam * 255
		quantized_np_secret_ddwt_cbam_num = np.clip(np_secret_ddwt_cbam, 0, 255).astype(np.uint8)

		image_array =  np.transpose(test_secrets_cbam.detach().cpu().numpy(), (0, 2, 3, 1))


		normalized_image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255

		normalized_image_array_16=normalized_image_array.astype(np.int16)
		normalized_image_array =normalized_image_array.astype(np.uint8)


		np_secret_cbam = np.transpose(test_secrets_cbam.detach().cpu().numpy(), (0, 2, 3, 1))

		np_secret_cbam = np_secret_cbam * 255
		quantized_secret_cbam_num = np.clip(np_secret_cbam, 0, 255).astype(np.uint8)



		emb_cover_num = (quantized_embeds - quantized_np_cover_cbam_num-quantized_np_secret_ddwt_cbam_num)

		secr_de_secr_num = (quantized_secret_num_16-normalized_image_array_16 )

		secr_de_secr_num = (secr_de_secr_num - np.min(secr_de_secr_num)) / (np.max(secr_de_secr_num) - np.min(secr_de_secr_num)) * 255

		secr_de_secr_num = secr_de_secr_num.squeeze()

		secr_de_secr_num = secr_de_secr_num.astype('uint8')





		diff_covers = (embeds_2 -embeds ) * 1000
		diff_secrets = (test_secrets_cbam - test_secrets) * 100000000000000

		cbam_cover_cov_num =np.transpose(diff_covers.detach().cpu().numpy(), (0, 2, 3, 1))
		cbam_cover_cov_num = cbam_cover_cov_num * 255
		cbam_cover_cov_num = np.clip(cbam_cover_cov_num, 0, 255).astype(np.uint8)

		cbam_secre_secret_num = np.transpose(diff_secrets.detach().cpu().numpy(), (0, 2, 3, 1))
		cbam_secre_secret_num = cbam_secre_secret_num * 255
		cbam_secre_secret_num = np.clip(cbam_secre_secret_num, 0, 255).astype(np.uint8)

		np_embed2 = np.transpose(embeds_2.detach().cpu().numpy(), (0, 2, 3, 1))

		np_embed2 = np_embed2 * 255
		quantized_embed2_num = np.clip(np_embed2, 0, 255).astype(np.uint8)


		for j in range(BATCH_SIZE):

			output_path1 = os.path.join(output_folder_de_secret, f'decode{i}_{j}_.tif')
			output_path2 = os.path.join(output_folder_cover, f'cover{i}_{j}.png')
			output_path3 = os.path.join(output_folder_secretpath, f'secret{i}_{j}.png')
			output_path4 = os.path.join(output_folder_quai_secretpath, f'qua_de_secret{i}_{j}.tif')

			output_path5 = os.path.join(output_folder_nosuofang_embed, f'nosuoemb{i}_{j}.png')
			output_path6 = os.path.join(output_folder_nosuofang_decode, f'nosuodecode{i}_{j}.tif')

			output_path7 = os.path.join(output_folder_emb_pad, f'nosuoemb{i}_{j}.png')
			output_path8 = os.path.join(output_folder_iwt_pad, f'nosuodecode{i}_{j}.png')
			output_path9 = os.path.join(output_folder_iwt, f'nosuoemb{i}_{j}.png')

			output_path10 = os.path.join(output_folder_cbam_cover, f'cover_cbam{i}_{j}.png')
			output_path11 = os.path.join(output_folder_cbam_secret, f'secret_cbam{i}_{j}.png')

			output_path12 = os.path.join(output_folder_emb_cove_20, f'emb_cover{i}_{j}.png')
			output_path13 = os.path.join(output_folder_desecert_secret_20, f'dese_secre{i}_{j}.png')
			output_path14 = os.path.join(output_folder_cbam_cover_cover, f'cbamcov_cov{i}_{j}.png')
			output_path15 = os.path.join(output_folder_cbam_secret_secret, f'cbase_secret{i}_{j}.png')

			output_path16 = os.path.join(output_folder_emb_2, f'emb2ed{i}_{j}.png')

			output_path17 = os.path.join(output_folder_emb_coverpre, f'emb2ed-precover{i}_{j}.png')




			cv2.imwrite(os.path.join(OUTPUT_PATH, f'embed{i}_{j}_min{min_np_embedsvalue}_max_{max_np_embedsvalue}.png'),
						cv2.cvtColor(quantized_embeds[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path1,
						cv2.cvtColor(quantized_outputs_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path2,
						cv2.cvtColor(quantized_cover_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path3,
						cv2.cvtColor(quantized_secret_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path4,
						cv2.cvtColor(quantized_de_secret_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path5,
						cv2.cvtColor(quantized_embeds_nosuofang[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path6,
						cv2.cvtColor(np_quantized_nosuofang_outputs22[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path7,
						cv2.cvtColor(np_qua_secret_pad_num [j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path8,
						cv2.cvtColor(np_qua_secret_pad_outputs_pad11[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path9,
						cv2.cvtColor(np_qua_secret_pad_outputs_pad[j], cv2.COLOR_RGB2BGR))

			cv2.imwrite(output_path10,
						cv2.cvtColor(quantized_np_cover_cbam_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path11,
						cv2.cvtColor(normalized_image_array[j], cv2.COLOR_RGB2BGR))

			cv2.imwrite(output_path12,
						cv2.cvtColor(emb_cover_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path13,
						cv2.cvtColor(secr_de_secr_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path14,
						cv2.cvtColor(cbam_cover_cov_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path15,
						cv2.cvtColor(cbam_secre_secret_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path16,
						cv2.cvtColor(quantized_embed2_num[j], cv2.COLOR_RGB2BGR))
			cv2.imwrite(output_path17,
						cv2.cvtColor(quantized_embed2_num[j], cv2.COLOR_RGB2BGR))





cover_mse /= i
secret_mse /= i
quantized_secret_mse /= i

ssim_emb_cov /= i
ssim_out_secret /= i
ssim_quati_secr /= i

psnr_emb_cov /= i
psnr_out_secret /= i
psnr_quai_secret /=i
psnr_emb_cov /= 2
psnr_out_secret /= 2
psnr_quai_secret /=2

print(f'Cover MSE: {cover_mse}')
print(f'Secret MSE: {secret_mse}')
print(f'Embed Quantized, Secret MSE: {quantized_secret_mse}')

print(f'Cover SSIM: {ssim_emb_cov}')
print(f'Secret SSIM: {ssim_out_secret}')
print(f'Embed Quantized, Secret SSIM: {ssim_quati_secr}')

print(f'Cover PSNR: {psnr_emb_cov}')
print(f'Secret PSNR: {psnr_out_secret}')
print(f'Embed Quantized, Secret PSNR: {psnr_quai_secret}')

