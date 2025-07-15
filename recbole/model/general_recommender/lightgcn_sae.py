# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
	Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
	https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.model.general_recommender.lightgcn import LightGCN




class LightGCN_SAE(LightGCN):
	r"""LightGCN is a GCN-based recommender model.

	LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
	collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
	propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
	learned at all layers as the final embedding.

	We implement the model following the original author with a pairwise training mode.
	"""

	input_type = InputType.PAIRWISE

	def __init__(self, config, dataset):
		super().__init__(config, dataset)
		model_path = config["base_path"]
		checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
		self.load_state_dict(checkpoint['state_dict'])
		self.sae_module = SAE(config)
		self.restore_item_e = None
		self.restore_user_e = None
		self.val_fvu = torch.tensor(0.0, device=self.device)
		for param in self.parameters():
			param.requires_grad = False

		for param in self.sae_module.parameters():
			param.requires_grad = True  


	def forward(self, train_mode=None):
		u_emb, i_emb = super().forward()
		u_emb_sae = (self.sae_module(u_emb, train_mode=train_mode))
		i_emb_sae = (self.sae_module(i_emb, train_mode=train_mode))
		return u_emb_sae, i_emb_sae
	
	def calculate_loss(self, interaction):
		if self.val_fvu.item() != 0:
			self.val_fvu = torch.tensor(0.0, device=self.device)
			self.recommendation_count = torch.zeros(self.n_items, dtype=torch.long, device=self.device)
		if self.restore_user_e is not None or self.restore_item_e is not None:
			self.restore_user_e, self.restore_item_e = None, None
		
		user_all_embeddings, item_all_embeddings = self.forward(train_mode=True)
		sae_loss = self.sae_module.fvu + self.sae_module.auxk_loss / 2
		self.sae_module.auxk_loss._fill(0.0)
		self.sae_module.fvu._fill(0.0)

		return sae_loss

	def full_sort_predict(self, interaction):
		user = interaction[self.USER_ID]
		if self.restore_user_e is None or self.restore_item_e is None:
			self.restore_user_e, self.restore_item_e = self.forward(train_mode=False)
		u_embeddings = self.restore_user_e[user]
		scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
		top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
		scores[:, 0] =  float("-inf")
		for key in top_recs.flatten():
			self.recommendation_count[key] += 1
		self.val_fvu += self.sae_module.fvu
		return scores.view(-1)
	
	def synthetic_inference(self, interaction, popular=None):
		user = interaction[self.USER_ID]
		if self.restore_user_e is None or self.restore_item_e is None:
			self.restore_user_e, self.restore_item_e = self.forward(train_mode=False)
		u_embeddings = self.restore_user_e[user]
		scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
		top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
		scores[:, 0] =  float("-inf")
		for key in top_recs.flatten():
			self.recommendation_count[key] += 1
		self.val_fvu += self.sae_module.fvu
		return scores.view(-1)


	def set_sae_mode(self, train_mode=True):
		self.sae_module.train_mode=train_mode
		

import torch
import numpy as np
import json
import torch
import torch.nn as nn
from recbole.utils import utils
import pandas as pd
import random

class SAE(nn.Module):
	
	def __init__(self,config):
		super(SAE, self).__init__()
		self.k = config["sae_k"]
		self.scale_size = config["sae_scale_size"]
		self.neuron_count = None
		self.unpopular_only = None
		self.corr_file = None
		self.device = config["device"]
		self.fvu = torch.tensor(0.0, device=self.device)
		self.auxk_loss = torch.tensor(0.0, device=self.device)
		self.dtype = torch.float32
		self.to(self.device)
		self.d_in = config['input_dim']
		self.hidden_dim = self.d_in * self.scale_size
		self.activation_count = torch.zeros(self.hidden_dim, device=config["device"])
		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
		self.encoder.bias.data.zero_()
		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
		self.set_decoder_norm_to_unit_norm()
		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))
		self.activate_latents = set()
		self.previous_activate_latents = None
		self.epoch_idx=0
		self.new_epoch = False
		self.item_activations = np.zeros(self.hidden_dim)
		self.highest_activations = {
			j: {
				"values": torch.empty(0, dtype=torch.float32, device=self.device),
				"low_values": torch.empty(0, dtype=torch.float32, device=self.device),
				"items": torch.empty(0, dtype=torch.long, device=self.device),
				"low_items": torch.empty(0, dtype=torch.long, device=self.device),
				"recommendations": torch.empty((0, 10), dtype=torch.long, device=self.device)
			}
			for j in range(self.hidden_dim)
		}
		return


	def set_dampen_hyperparam(self, corr_file=None, N=None, beta=None, unpopular_only=False):
		self.corr_file = corr_file
		self.N = N
		self.beta = beta
		self.unpopular_only = unpopular_only	
  
  
	def get_dead_latent_ratio(self, need_update=0):
		# Calculate the dead latent ratio
		ans = 1 - len(self.activate_latents) / self.hidden_dim
		# Calculate the current number of dead latents
		current_dead = self.hidden_dim - len(self.activate_latents)
		print("Dead percentage: ", ans)
		print(f"FVU: {self.fvu}, AUXK Loss: {self.auxk_loss}, AUXK Loss / 2: {self.auxk_loss / 2} SAE Total Loss: {(self.auxk_loss / 2) + self.fvu}")
		if need_update:
			# Convert current active latents to a tensor
			current_active = torch.tensor(list(self.activate_latents), device=self.device)
			
			# Compute revived latents if there’s a previous state
			if self.previous_activate_latents is not None:
				# Find latents in current_active that were not in previous_activate_latents
				revived_mask = ~torch.isin(current_active, self.previous_activate_latents)
				num_revived = revived_mask.sum().item()
				# Print the requested information
				print(f"Number of revived latents: {num_revived}, Current dead latents: {current_dead}")
			
			# Update previous_activate_latents to the current active latents
			self.previous_activate_latents = current_active
		
			# Reset activate_latents for the next period
			self.activate_latents = set()
		return ans


	def set_decoder_norm_to_unit_norm(self):
		assert self.W_dec is not None, "Decoder weight was not initialized."
		eps = torch.finfo(self.W_dec.dtype).eps
		norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
		self.W_dec.data /= norm + eps


	def topk_activation(self, x, sequences, save_result, k=0):
		"""
		Performs top-k activation on tensor x.
		If k is not None, reads the first k indices from the previously saved indices file
		and sets their activations in x to -10 before computing top-k.
		Returns a sparse tensor with only the top-k activations.
		"""

		topk_values, topk_indices = torch.topk(x, self.k, dim=1)
		flat_indices = topk_indices.view(-1)

		# Count occurrences of each index
		counts = torch.bincount(flat_indices, minlength=self.hidden_dim)

		# Update activation count
		self.activation_count += counts.to(self.activation_count.device)
		self.activate_latents.update(topk_indices.cpu().numpy().flatten())

		# Save epoch activations if needed
		if save_result:
			values_np = topk_values.detach().cpu().numpy()
			inds_np = topk_indices.detach().cpu().numpy()
		# Build sparse output
		sparse_x = torch.zeros_like(x)
		sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
		return sparse_x

		

	def update_topk_recommendations(self, predictions, current_sequences, k=10):
		"""
		Update top-k recommendations for sequences in highest_activations.

		Parameters:
		- predictions: Tensor of shape [B, N], where B is batch size and N is the number of items.
		- current_sequences: List of sequences (IDs) in the current batch.
		- k: Number of top recommendations to save.
		"""
		# Convert current_sequences to a list of lists for easy comparison
		current_sequences_list = [seq.tolist() for seq in current_sequences]
  
		for neuron_idx, data in self.highest_activations.items():
			for idx, stored_sequence in enumerate(data["sequences"]):
				# Check if the stored sequence is in the current batch
				if stored_sequence in current_sequences_list:
					# Find the index of the stored sequence in the current batch
					batch_idx = current_sequences_list.index(stored_sequence)
					
					# Get predictions for this sequence
					pred_scores = predictions[batch_idx].cpu().numpy()  # Convert to numpy for sorting
					
					# Find indices of the top-k scores
					topk_indices = np.argsort(pred_scores, axis=1)[:, -k:][:, ::-1]  # Add 1 to match item IDs
	 
					# Update the recommendations for this sequence
					data["recommendations"].append(topk_indices.tolist())
	 


	def dampen_neurons(self, pre_acts, dataset=None):
		if self.N is None or self.N==0:
			return pre_acts
		pop_neurons, unpop_neurons = utils.get_extreme_correlations(self.corr_file, self.unpopular_only, dataset=dataset)
  
		# Combine both groups into one list while labeling the group type.
		# 'unpop' neurons are those with higher activations for unpopular inputs (to be reinforced),
		# while 'pop' neurons are those with lower activations (to be dampened).
		combined_neurons = [(idx, cohen, 'unpop') for idx, cohen in unpop_neurons] + \
						[(idx, cohen, 'pop') for idx, cohen in pop_neurons]
						
		# Now sort by the absolute Cohen's d value (in descending order) and pick the overall top N neurons.
		combined_sorted = sorted(combined_neurons, key=lambda x: abs(x[1]), reverse=True)
		top_neurons = combined_sorted[:int(self.N)]
		# Load the corresponding statistics files.
		stats_unpop = pd.read_csv(rf"./dataset/{dataset}/row_stats_popular.csv")
		stats_pop = pd.read_csv(rf"./dataset/{dataset}/row_stats_unpopular.csv")
  
		# Create tensors of the absolute Cohen's d values for the selected neurons.
		abs_cohens = torch.tensor([abs(c) for _, c, _ in top_neurons], device=pre_acts.device)
		min_cohen = min(abs(c) for _, c, _ in combined_neurons)
		# Define a helper normalization function.
		def normalize_to_range(x, new_min, new_max):
			min_val = min_cohen
			max_val = torch.max(x)
			if max_val == min_val:
				return torch.full_like(x, (new_min + new_max) / 2)
			return (x - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

		# Normalize the Cohen's d values to [0, 2.5]
		weights = normalize_to_range(abs_cohens, new_min=0, new_max=self.beta)

		# Now update the neuron activations based on group.
		for i, (neuron_idx, cohen, group) in enumerate(top_neurons):
			weight = weights[i]		
			if group == 'unpop':
				# For neurons to be reinforced, fetch stats from the unpopular file.
				row = stats_unpop.iloc[neuron_idx]
				mean_val = row["mean"]
				std_val = row["std"]
	
				# Identify positions where the neuron's activation is above its mean.
				vals = pre_acts[:, neuron_idx]
				# Increase activations by an amount proportional to the standard deviation and effective weight.
				pre_acts[:, neuron_idx] += weight * std_val
			else:  # group == 'pop'
				# For neurons to be dampened, use the popular statistics for impact.
				pop_mean = stats_pop.iloc[neuron_idx]["mean"]
				pop_sd = stats_pop.iloc[neuron_idx]["std"]

				# Still fetch the comparison stats from the unpopular stats file
				# (this is from your original logic; adjust if needed).
				row = stats_unpop.iloc[neuron_idx]
				mean_val = row["mean"]
				std_val = row["std"]

				# Identify positions where the neuron's activation is below its mean.
				vals = pre_acts[:, neuron_idx]
				# Decrease activations proportionally.
				pre_acts[:, neuron_idx] -= weight * pop_sd
	
		return pre_acts
	 
	 
	
	def add_noise(self, pre_acts, std):
		pre_actss = pre_acts.detach().cpu()
		if self.N is None:
			return pre_acts

		# pick N unique neurons
		top_neurons = random.sample(range(self.hidden_dim), int(self.N))

		# add Gaussian noise to each selected neuron
		# pre_acts shape: (batch_size, hidden_dim)
		batch_size = pre_actss.shape[0]
		for idx in top_neurons:
			# draw a vector of Gaussian noise
			noise = np.random.normal(
				loc=0.0,
				scale=std,
				size=(batch_size,)
			)
			pre_actss[:, idx] += noise

		return pre_actss.to(self.device)
	 
	 
	def forward(self, x, sequences=None, train_mode=False, save_result=False, epoch=None, dataset=None, pop_scores=None):
		
			sae_in = x - self.b_dec
			pre_acts = self.encoder(sae_in)
			self.last_activations = pre_acts
			if self.corr_file:
				pre_acts = self.dampen_neurons(pre_acts, dataset=dataset)
				# pre_acts = self.add_noise(pre_acts, std=self.beta)
			pre_acts = nn.functional.relu(pre_acts)   
			z = self.topk_activation(pre_acts, sequences, save_result=False)

			x_reconstructed = z @ self.W_dec + self.b_dec
			e = x_reconstructed - x
			total_variance = (x - x.mean(0)).pow(2).sum()
			self.fvu += e.pow(2).sum() / total_variance

			if train_mode:
				if self.new_epoch == True:
					self.new_epoch = False
					dead = self.get_dead_latent_ratio(need_update=1)
					print("Dead percentage ", dead)					
				# First epoch, do not have dead latent info
				if self.previous_activate_latents is None:
					self.auxk_loss._fill(0.0)
					return x_reconstructed
				num_dead = self.hidden_dim - len(self.previous_activate_latents)
				k_aux = int(x.shape[-1]) * 2
				if num_dead == 0:
					self.auxk_loss._fill(0.0)
					return x_reconstructed
				scale = min(num_dead / k_aux, 1.0)
				k_aux = min(k_aux, num_dead)
				dead_mask = torch.isin(
					torch.arange(pre_acts.shape[-1]).to(self.device),
					self.previous_activate_latents,
					invert=True
				)
				auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
				auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
				# print("these are aux values, ", auxk_indices[0])
				# print("these are aux indices, ", auxk_acts[0])

				e_hat = torch.zeros_like(auxk_latents)
				e_hat.scatter_(1, auxk_indices, auxk_acts.to(self.dtype))
				e_hat = e_hat @ self.W_dec + self.b_dec

				auxk_loss = (e_hat - e).pow(2).sum()
				self.auxk_loss += scale * auxk_loss / total_variance

			return x_reconstructed


