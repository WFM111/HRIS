"""

DEPRECATED Usage:
mkdir ckpt_dir
python3 train_baseline.py tiny-imagenet-200 ckpt_dir none [restart ckpt]
python3 train_baseline.py tiny-imagenet-200 ckpt_dir corr [restart ckpt]
Sources:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
from itertools import chain
import os, argparse
import sys, time


import torch
import torch.nn.functional as F
import pytorch_msssim
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from model import Stegnet

from model import  ResidualDenseBlock_out_drop
from model import ResidualBlockNoBN2



from loss import correlation
from tiramisu import FCDenseNet


from resnet_cbam import CBAM_2MLP


import numpy as np
import pywt



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def round_diff(x):
    sign = torch.ones_like(x)
    sign[torch.floor(x) % 2 == 0] = -1
    y = sign * torch.cos(x * torch.pi) / 2
    out = torch.round(x) + y - y.detach()
    return out
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


class Training():
	def __init__(self, batch_size, epochs):
		args = self.load_args()

		self.train_path=args.train_set

		self.val_path=args.val_set


		output_path = Path(args.output_dir)
		loss_type = args.loss
		self.load_weights = args.load_weights

		self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) if not self.load_weights else self.load_weights

		self.corr_loss = correlation if loss_type == 'corr' else None

		self.batch_size = batch_size
		self.epochs = epochs

		if self.load_weights:


			self.output_path = output_path/self.load_weights
			print(self.output_path)

			if not self.output_path.is_dir(): raise ValueError('Checkpoint not found.')
		else:
			self.output_path =output_path/self.timestamp
			if not self.output_path.is_dir(): self.output_path.mkdir(parents=True)
		#  tensorboard logger
		self.writer = SummaryWriter(log_dir = self.output_path/'callbacks')

	def load_args(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--train_set', type=str, help='Path to train_data.', default='D:\\code_yinxie\\811data\\train')

		parser.add_argument('--val_set', type=str, help='Path to val_data.', default='D:\\code_yinxie\\811data\\val')

		parser.add_argument('--output_dir', type=str, help='Path to output.',default='a_CBAM2MLP_ssconv3__dendrop_semode' )

		parser.add_argument('--loss', type=str, help='Loss to use.', default='none')
		parser.add_argument('--load_weights', type=str, help='Restart from checkpoint with timestamp.',required=False)
		args = parser.parse_args()
		return args
	
	def data_loader(self):
		data_transform = T.Compose([

			T.CenterCrop((256, 256)),
			T.ToTensor(),

		])


		train_cover_dataset = datasets.ImageFolder(os.path.join(self.train_path, 'cover'), transform=data_transform)
		train_secret_dataset = datasets.ImageFolder(os.path.join(self.train_path, 'secret'), transform=data_transform)
		val_cover_dataset = datasets.ImageFolder(os.path.join(self.val_path, 'cover'), transform=data_transform)
		val_secret_dataset = datasets.ImageFolder(os.path.join(self.val_path, 'secret'), transform=data_transform)

		train_cover_dataloader = DataLoader(train_cover_dataset, batch_size=self.batch_size * 2,
								shuffle=True, drop_last=True)
		train_secret_dataloader = DataLoader(train_secret_dataset, batch_size=self.batch_size * 2,
								shuffle=True, drop_last=True)
		val_cover_dataloader = DataLoader(val_cover_dataset, batch_size=self.batch_size * 2,
							shuffle=False, drop_last=True)
		val_secret_dataloader = DataLoader(val_secret_dataset, batch_size=self.batch_size * 2,
							shuffle=False, drop_last=True)
		return train_cover_dataloader, train_secret_dataloader,val_cover_dataloader,val_secret_dataloader
	
	def load_ckpts(self,resnet, postEnhance, optimizer):
		print(f'Loading weights from {self.output_path}')
		all_weights = []
		for file in os.listdir(self.output_path/'weights'):
			if file.endswith('.pt'):
				all_weights.append(file)
		latest_weights = sorted(all_weights)[-1]
		checkpoint = torch.load(self.output_path/'weights'/latest_weights, map_location='cpu')

		postEnhance.load_state_dict(checkpoint['postEnhance_dict'])

		optimizer.load_state_dict(checkpoint['optimizer_rehance_dict'])
		resnet.load_state_dict(checkpoint['resnet_dict'])
		return resnet, postEnhance, optimizer, len(all_weights)



	def train(self, train_cover_dataloader,train_secret_dataloader,val_cover_dataloader,val_secret_dataloader):

		precoder=CBAM_2MLP().to(DEVICE)


		resnet = ResidualBlockNoBN2().to(DEVICE)

		postEnhance = ResidualDenseBlock_out_drop().to(DEVICE)

		encoder = FCDenseNet().to(DEVICE)
		decoder = Stegnet(3).to(DEVICE)
		WEIGHT_PATH1= 'D:\\code_yinxie\\Stegnet-Covertcast-main_torch\\cbam_2mlp\\epoch-100.pt'
		checkpoint = torch.load(WEIGHT_PATH1, map_location='cpu')

		encoder.load_state_dict(checkpoint['encoder_dict'])

		decoder.load_state_dict(checkpoint['decoder_dict'])
		precoder.load_state_dict(checkpoint['precoder_dict'])
		for param in precoder.parameters():
			param.requires_grad = False
		for param in encoder.parameters():
			param.requires_grad = False
		for param in decoder.parameters():
			param.requires_grad = False


		optimizer = torch.optim.Adam(chain(resnet.parameters(),postEnhance.parameters()))
		start_epoch = 0
		if self.load_weights:
			resnet,postEnhance, optimizer, start_epoch = self.load_ckpts(resnet,postEnhance, optimizer)

		for epoch in range(start_epoch, self.epochs):

			precoder.train()
			encoder.train()
			decoder.train()
			resnet.train()
			postEnhance.train()
			train_loss = 0.0
			for i, (train_cover_data,train_secret_data) in tqdm(enumerate(zip(train_cover_dataloader, train_secret_dataloader), start=1)):

				train_cover_inputs, _ = train_cover_data
				train_secret_inputs, _ =train_secret_data

				train_cover = train_cover_inputs.to(DEVICE)
				train_secret=train_secret_inputs.to(DEVICE)
				train_cover_cbam = precoder(train_cover)
				train_secret_cbam = precoder(train_secret)
				train_secret_dwt = dwt_init(train_secret_cbam,DEVICE)
				optimizer.zero_grad()
				top_padding_size = 8
				train_cover_1 = F.pad(train_cover_cbam,
									  (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
									  mode='reflect')
				train_secret_1 = F.pad(train_secret_dwt,
									   (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
									   mode='reflect')
				original_height, original_width = train_secret_1.shape[2], train_secret_1.shape[3]
				embeds = encoder(torch.cat([train_cover_1,train_secret_1], dim=1))
				top_crop = top_padding_size
				bottom_crop = original_height - top_padding_size
				left_crop = top_padding_size
				right_crop = original_width - top_padding_size
				embeds = embeds[:, :, top_crop:bottom_crop, left_crop:right_crop]

				np_embeds = embeds * 255
				min_np_embedsvalue = int(torch.min(np_embeds).item()) - 1
				max_np_embedsvalue = int(torch.max(np_embeds).item()) + 1

				np_embeds = ((np_embeds - min_np_embedsvalue) / (max_np_embedsvalue - min_np_embedsvalue) * 255).float()
				np_embeds =round_diff(np_embeds)
				np_embeds = torch.clamp(np_embeds, 0, 255)


				quantized_tensor = (np_embeds.float() / 255) * (
						max_np_embedsvalue - min_np_embedsvalue) + min_np_embedsvalue

				quantized_tensor1 = quantized_tensor / 255
				quantized_tensor1=resnet(quantized_tensor1)
				outputs_dwt = decoder(quantized_tensor1)
				outputs_dwt = postEnhance(outputs_dwt)
				outputs =  iwt_init(outputs_dwt,DEVICE)
				loss = (F.l1_loss(embeds, train_cover) + F.l1_loss(outputs,train_secret) +F.l1_loss(outputs_dwt,train_secret_dwt) +F.l1_loss(embeds,quantized_tensor1) +

						torch.mean(torch.var(embeds - train_cover, dim=(1, 2, 3), unbiased=True)) +
						torch.mean(torch.var(outputs_dwt-train_secret_dwt, dim=(1, 2, 3), unbiased=True))  +

						torch.mean(torch.var(embeds - quantized_tensor1, dim=(1, 2, 3), unbiased=True)) +
						torch.mean(torch.var(outputs - train_secret, dim=(1, 2, 3), unbiased=True)))
				if self.corr_loss is not None:
					loss += torch.abs(self.corr_loss(embeds - train_cover, train_secret))+torch.abs(self.corr_loss(outputs- train_secret, train_cover))

				loss.backward()
				optimizer.step()
				train_loss += loss.item()
			train_loss /= i

			weight_path = self.output_path/"weights"
			weight_path.mkdir(exist_ok=True)
			torch.save({
				'epoch': epoch,
				'loss': train_loss,

				'resnet_dict': resnet.state_dict(),
				'postEnhance_dict': postEnhance.state_dict(),

				'optimizer_rehance_dict': optimizer.state_dict()
			}, weight_path/f"epoch-{epoch}_enhance.pt")

			self.writer.add_scalar('Loss/train', train_loss, epoch)

			precoder.eval()
			resnet.eval()
			encoder.eval()
			decoder.eval()
			postEnhance.eval()
			with torch.no_grad():

				val_loss = 0.0
				for i, (val_cover_data,val_secret_data) in tqdm(enumerate(zip(val_cover_dataloader,val_secret_dataloader), start=1)):

					val_cover_inputs, _ = val_cover_data
					val_secret_inputs, _ =val_secret_data
					val_cover = val_cover_inputs.to(DEVICE)
					val_secret=val_secret_inputs.to(DEVICE)

					val_cover_cbam = precoder(val_cover)
					val_secret_cbam = precoder(val_secret)
					val_secret_dwt = dwt_init(val_secret_cbam , DEVICE)
					top_padding_size = 8
					val_cover_1 = F.pad(val_cover_cbam,
										  (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
										  mode='reflect')
					val_secret_1 = F.pad(val_secret_dwt,
										   (top_padding_size, top_padding_size, top_padding_size, top_padding_size),
										   mode='reflect')

					original_height, original_width = val_secret_1.shape[2], val_secret_1.shape[3]

					embeds = encoder(torch.cat([val_cover_1, val_secret_1], dim=1))
					top_crop = top_padding_size
					bottom_crop = original_height - top_padding_size
					left_crop = top_padding_size
					right_crop = original_width - top_padding_size

					embeds = embeds[:, :, top_crop:bottom_crop, left_crop:right_crop]

					np_embeds = embeds * 255


					min_np_embedsvalue = int(torch.min(np_embeds).item()) - 1
					max_np_embedsvalue = int(torch.max(np_embeds).item()) + 1


					np_embeds = ((np_embeds - min_np_embedsvalue) / (max_np_embedsvalue - min_np_embedsvalue) * 255).float()
					np_embeds = round_diff(np_embeds)
					np_embeds = torch.clamp(np_embeds, 0, 255)


					quantized_tensor = (np_embeds.float() / 255) * (
							max_np_embedsvalue - min_np_embedsvalue) + min_np_embedsvalue

					quantized_tensor1 = quantized_tensor / 255

					quantized_tensor1=resnet(quantized_tensor1)
					outputs_dwt = decoder(quantized_tensor1)

					outputs = iwt_init(outputs_dwt,DEVICE)
					outputs =postEnhance(outputs)

					loss = (F.l1_loss(embeds, val_cover) + F.l1_loss(outputs, val_secret) +F.l1_loss(outputs_dwt, val_secret_dwt)+F.l1_loss(embeds,quantized_tensor1) +

							torch.mean(torch.var(embeds - val_cover, dim=(1, 2, 3), unbiased=True)) +
							torch.mean(torch.var(embeds - quantized_tensor1, dim=(1, 2, 3), unbiased=True)) +

							torch.mean(torch.var(outputs_dwt - val_secret_dwt, dim=(1, 2, 3), unbiased=True)) +
							torch.mean(torch.var(outputs - val_secret, dim=(1, 2, 3), unbiased=True)))

					if self.corr_loss is not None:
						loss += torch.abs(self.corr_loss(embeds - val_cover, val_secret)) + torch.abs(
							self.corr_loss(outputs - val_secret, val_cover))
					val_loss += loss.item()
				val_loss /= i
			self.writer.add_scalar('Loss/val', val_loss, epoch)

			print(f"epoch={epoch};train_loss={train_loss};val_loss={val_loss}")
			self.writer.close()


if __name__ == '__main__':
	batch_size = 1

	epochs =100
	train = Training(batch_size, epochs)
	train_cover_dataloader,train_secret_dataloader,val_cover_dataloader,val_secret_dataloader = train.data_loader()
	train.train(train_cover_dataloader,train_secret_dataloader,val_cover_dataloader,val_secret_dataloader)

