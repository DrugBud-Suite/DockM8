import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys, tempfile, shutil
import MDAnalysis as mda
sys.path.append(os.path.abspath(__file__).replace("rtmscore.py",".."))
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F
import dgl
import random
import dgl.function as fn
from torch import nn

import torch.nn as nn
import random
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr
import dgl

import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset #, DataLoader    
from rdkit import Chem
from joblib import Parallel, delayed

import re, os
import dgl
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl.data.utils import save_graphs, load_graphs, load_labels
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances



def scoring(prot, lig, modpath,
			cut=10.0,
			gen_pocket=False,
			reflig=None,
			atom_contribution=False,
			res_contribution=False,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			**kwargs
			):
	"""
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	reflig: The reference ligand to determine the pocket.
	cut: The distance within the reference ligand to determine the pocket.
	atom_contribution: whether the decompose the score at atom level.
	res_contribution: whether the decompose the score at residue level.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.	
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""
	#try:
	data = VSDataset(ligs=lig,
					prot=prot,
					cutoff=cut,		
					gen_pocket=gen_pocket,
					reflig=reflig,
					explicit_H=explicit_H, 
					use_chirality=use_chirality,
					parallel=parallel)
					
						
	test_loader = DataLoader(dataset=data, 
							batch_size=kwargs["batch_size"],
							shuffle=False, 
							num_workers=kwargs["num_workers"],
							collate_fn=collate)
	
	ligmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsl"], 
									edge_features=kwargs["num_edge_featsl"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
	
	protmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsp"], 
									edge_features=kwargs["num_edge_featsp"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
						
	model = RTMScore(ligmodel, protmodel, 
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])
	
	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict']) 
	if atom_contribution:
		preds, at_contrs, _ = run_an_eval_epoch(model, 
												test_loader, 
												pred=True, 
												atom_contribution=True, 
												res_contribution=False, 
												dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
		
		atids = ["%s%s"%(a.GetSymbol(),a.GetIdx()) for a in data.ligs[0].GetAtoms()]
		return data.ids, preds, atids, at_contrs
	
	elif res_contribution:
		preds, _, res_contrs = run_an_eval_epoch(model, 
												test_loader, 
												pred=True, 
												atom_contribution=False, 
												res_contribution=True, 
												dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
		u = mda.Universe(data.prot)
		resids = ["%s_%s%s"%(x[0],y,z) for x,y,z in zip(u.residues.chainIDs, u.residues.resnames, u.residues.resids)]
		return data.ids, preds, resids, res_contrs
	else:	
		preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
		return data.ids, preds


def rtmscore(prot, lig, output, model, ncpus, gen_pocket=False, reflig=None, cutoff=10.0, parallel=True, atom_contribution=False, res_contribution=False):
	args={}
	args["batch_size"] = 128
	args["dist_threhold"] = 5
	args['device'] = 'cpu'
	args["num_workers"] = ncpus
	args["num_node_featsp"] = 41
	args["num_node_featsl"] = 41
	args["num_edge_featsp"] = 5
	args["num_edge_featsl"] = 10
	args["hidden_dim0"] = 128
	args["hidden_dim"] = 128 
	args["n_gaussians"] = 10
	args["dropout_rate"] = 0.10
	if atom_contribution:
		ids, scores, atids, at_contrs = scoring(prot=prot, 
											lig=lig, 
											modpath=model,
											cut=cutoff,
											gen_pocket=gen_pocket,
											reflig=reflig,
											atom_contribution=True,
											explicit_H=False, 
											use_chirality=True,
											parallel=parallel,
											**args
											)
		df = pd.DataFrame(at_contrs).T
		df.columns= ids
		df.index = atids
		df = df[df.apply(np.sum,axis=1)!=0].T
		dfx = pd.DataFrame(zip(*(ids, scores)),columns=["Pose ID","RTMScore"])
		dfx.index = dfx.id
		df = pd.concat([dfx["RTMScore"],df],axis=1)
		df.sort_values("RTMScore", ascending=False, inplace=True)	
		df.to_csv(output)
		return df
	elif res_contribution:
		ids, scores, resids, res_contrs = scoring(prot=prot, 
											lig=lig, 
											modpath=model,
											cut=cutoff,
											gen_pocket=gen_pocket,
											reflig=reflig,
											res_contribution=True,
											explicit_H=False, 
											use_chirality=True,
											parallel=parallel,
											**args
											)
		df = pd.DataFrame(res_contrs).T
		df.columns= ids
		df.index = resids
		df = df[df.apply(np.sum,axis=1)!=0].T
		dfx = pd.DataFrame(zip(*(ids, scores)),columns=["Pose ID","RTMScore"])
		dfx.index = dfx.id
		df = pd.concat([dfx["RTMScore"],df],axis=1)
		df.sort_values("RTMScore", ascending=False, inplace=True)	
		df.to_csv(output)	
		return df		
	else:
		ids, scores = scoring(prot=prot, 
							lig=lig, 
							modpath=model,
							cut=cutoff,
							gen_pocket=gen_pocket,
							reflig=reflig,
							explicit_H=False, 
							use_chirality=True,
							parallel=parallel,
							**args
							)
		df = pd.DataFrame(zip(*(ids, scores)),columns=["Pose ID","RTMScore"])
		df.sort_values("RTMScore", ascending=False, inplace=True)
		df.to_csv(output, index=False)
		return df

##
#the model architecture of graph transformer is modified from https://github.com/BioinfoMachineLearning/DeepInteract

def glorot_orthogonal(tensor, scale):
	"""Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
	if tensor is not None:
		th.nn.init.orthogonal_(tensor.data)
		scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
		tensor.data *= scale.sqrt()


class MultiHeadAttentionLayer(nn.Module):
	"""Compute attention scores with a DGLGraph's node and edge (geometric) features."""
	def __init__(self, num_input_feats, num_output_feats,
				num_heads, using_bias=False, update_edge_feats=True):
		super(MultiHeadAttentionLayer, self).__init__()
		
        # Declare shared variables
		self.num_output_feats = num_output_feats
		self.num_heads = num_heads
		self.using_bias = using_bias
		self.update_edge_feats = update_edge_feats
		
		# Define node features' query, key, and value tensors, and define edge features' projection tensors
		self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		
		self.reset_parameters()
		
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		scale = 2.0
		if self.using_bias:
			glorot_orthogonal(self.Q.weight, scale=scale)
			self.Q.bias.data.fill_(0)
			
			glorot_orthogonal(self.K.weight, scale=scale)
			self.K.bias.data.fill_(0)
			
			glorot_orthogonal(self.V.weight, scale=scale)
			self.V.bias.data.fill_(0)
			
			glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
			self.edge_feats_projection.bias.data.fill_(0)
		else:
			glorot_orthogonal(self.Q.weight, scale=scale)
			glorot_orthogonal(self.K.weight, scale=scale)
			glorot_orthogonal(self.V.weight, scale=scale)
			glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
	
	def propagate_attention(self, g):
		# Compute attention scores
		g.apply_edges(lambda edges: {"score": edges.src['K_h'] * edges.dst['Q_h']})
		# Scale and clip attention scores
		g.apply_edges(lambda edges: {"score": (edges.data["score"]/np.sqrt(self.num_output_feats)).clamp(-5.0,5.0)})		
		# Use available edge features to modify the attention scores
		g.apply_edges(lambda edges: {"score": edges.data['score'] * edges.data['proj_e']})
		# Copy edge features as e_out to be passed to edge_feats_MLP
		if self.update_edge_feats:
			g.apply_edges(lambda edges: {"e_out": edges.data["score"]})
		
		# Apply softmax to attention scores, followed by clipping
		g.apply_edges(lambda edges: {"score": th.exp((edges.data["score"].sum(-1, keepdim=True)).clamp(-5.0,5.0))})
		# Send weighted values to target nodes
		#e_ids = g.edges()
		#g.send_and_recv(e_ids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
		#g.send_and_recv(e_ids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
		g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
		g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'))
	
	def forward(self, g, node_feats, edge_feats):
		with g.local_scope():
			e_out = None
			node_feats_q = self.Q(node_feats)
			node_feats_k = self.K(node_feats)
			node_feats_v = self.V(node_feats)
			edge_feats_projection = self.edge_feats_projection(edge_feats)			
			# Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
			g.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
			g.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
			g.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
			g.edata['proj_e'] = edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)
			# Disperse attention information
			self.propagate_attention(g)
			# Compute final node and edge representations after multi-head attention
			h_out = g.ndata['wV'] / (g.ndata['z'] + th.full_like(g.ndata['z'], 1e-6))  # Add eps to all
			if self.update_edge_feats:
				e_out = g.edata['e_out']
		# Return attention-updated node and edge representations
		return h_out, e_out


class GraphTransformerModule(nn.Module):
	"""A Graph Transformer module (equivalent to one layer of graph convolutions)."""
	def __init__(
			self,
			num_hidden_channels,
			activ_fn=nn.SiLU(),
			residual=True,
			num_attention_heads=4,
			norm_to_apply='batch',
			dropout_rate=0.1,
			num_layers=4,
			):
		super(GraphTransformerModule, self).__init__()
		
		# Record parameters given
		self.activ_fn = activ_fn
		self.residual = residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
		
		self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
		if self.apply_layer_norm:
			self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.mha_module = MultiHeadAttentionLayer(
			self.num_hidden_channels,
			self.num_output_feats // self.num_attention_heads,
			self.num_attention_heads,
			self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
			update_edge_feats=True
		)
		
		self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		
		# MLP for node features
		dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
		self.node_feats_MLP = nn.ModuleList([
			nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
			self.activ_fn,
			dropout,
			nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
		])
		
		if self.apply_layer_norm:
			self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		# MLP for edge features
		self.edge_feats_MLP = nn.ModuleList([
			nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
			self.activ_fn,
			dropout,
			nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
		])
		
		self.reset_parameters()
	
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		scale = 2.0
		glorot_orthogonal(self.O_node_feats.weight, scale=scale)
		self.O_node_feats.bias.data.fill_(0)
		glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
		self.O_edge_feats.bias.data.fill_(0)
		
		for layer in self.node_feats_MLP:
			if hasattr(layer, 'weight'):  # Skip initialization for activation functions
				glorot_orthogonal(layer.weight, scale=scale)
		
		for layer in self.edge_feats_MLP:
			if hasattr(layer, 'weight'):
				glorot_orthogonal(layer.weight, scale=scale)
	
	def run_gt_layer(self, g, node_feats, edge_feats):
		"""Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
		node_feats_in1 = node_feats  # Cache node representations for first residual connection
		edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection
			
		# Apply first round of normalization before applying graph attention, for performance enhancement
		if self.apply_layer_norm:
			node_feats = self.layer_norm1_node_feats(node_feats)
			edge_feats = self.layer_norm1_edge_feats(edge_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm1_node_feats(node_feats)
			edge_feats = self.batch_norm1_edge_feats(edge_feats)
		
		# Get multi-head attention output using provided node and edge representations
		node_attn_out, edge_attn_out = self.mha_module(g, node_feats, edge_feats)
		
		node_feats = node_attn_out.view(-1, self.num_output_feats)
		edge_feats = edge_attn_out.view(-1, self.num_output_feats)
		
		node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
		edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)
		
		node_feats = self.O_node_feats(node_feats)
		edge_feats = self.O_edge_feats(edge_feats)
		
		# Make first residual connection
		if self.residual:
			node_feats = node_feats_in1 + node_feats  # Make first node residual connection
			edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection
		
		node_feats_in2 = node_feats  # Cache node representations for second residual connection
		edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection
		
		# Apply second round of normalization after first residual connection has been made
		if self.apply_layer_norm:
			node_feats = self.layer_norm2_node_feats(node_feats)
			edge_feats = self.layer_norm2_edge_feats(edge_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm2_node_feats(node_feats)
			edge_feats = self.batch_norm2_edge_feats(edge_feats)
		
		# Apply MLPs for node and edge features
		for layer in self.node_feats_MLP:
			node_feats = layer(node_feats)
		for layer in self.edge_feats_MLP:
			edge_feats = layer(edge_feats)
		
		# Make second residual connection
		if self.residual:
			node_feats = node_feats_in2 + node_feats  # Make second node residual connection
			edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection
		
		# Return edge representations along with node representations (for tasks other than interface prediction)
		return node_feats, edge_feats
	
	def forward(self, g, node_feats, edge_feats):
		"""Perform a forward pass of a Graph Transformer to get intermediate node and edge representations."""
		node_feats, edge_feats = self.run_gt_layer(g, node_feats, edge_feats)
		return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
	"""A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""	
	def __init__(self,
				num_hidden_channels,
				activ_fn=nn.SiLU(),
				residual=True,
				num_attention_heads=4,
				norm_to_apply='batch',
				dropout_rate=0.1,
				num_layers=4):
		super(FinalGraphTransformerModule, self).__init__()
		
		# Record parameters given
		self.activ_fn = activ_fn
		self.residual = residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
		
		self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
		if self.apply_layer_norm:
			self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.mha_module = MultiHeadAttentionLayer(
					self.num_hidden_channels,
					self.num_output_feats // self.num_attention_heads,
					self.num_attention_heads,
					self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
					update_edge_feats=False)
		
		self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		
		# MLP for node features
		dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
		self.node_feats_MLP = nn.ModuleList([
					nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
					self.activ_fn,
					dropout,
					nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
					])
		
		if self.apply_layer_norm:
			self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.reset_parameters()
	
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		scale = 2.0
		glorot_orthogonal(self.O_node_feats.weight, scale=scale)
		self.O_node_feats.bias.data.fill_(0)
		
		for layer in self.node_feats_MLP:
			if hasattr(layer, 'weight'):  # Skip initialization for activation functions
				glorot_orthogonal(layer.weight, scale=scale)
		
		#glorot_orthogonal(self.conformation_module.weight, scale=scale)
	
	def run_gt_layer(self, g, node_feats, edge_feats):
		"""Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
		node_feats_in1 = node_feats  # Cache node representations for first residual connection
		
		# Apply first round of normalization before applying graph attention, for performance enhancement
		if self.apply_layer_norm:
			node_feats = self.layer_norm1_node_feats(node_feats)
			edge_feats = self.layer_norm1_edge_feats(edge_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm1_node_feats(node_feats)
			edge_feats = self.batch_norm1_edge_feats(edge_feats)
		
		# Get multi-head attention output using provided node and edge representations
		node_attn_out, _ = self.mha_module(g, node_feats, edge_feats)
		node_feats = node_attn_out.view(-1, self.num_output_feats)		
		node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
		node_feats = self.O_node_feats(node_feats)
		
		# Make first residual connection
		if self.residual:
			node_feats = node_feats_in1 + node_feats  # Make first node residual connection
		
		node_feats_in2 = node_feats  # Cache node representations for second residual connection
		
		# Apply second round of normalization after first residual connection has been made
		if self.apply_layer_norm:
			node_feats = self.layer_norm2_node_feats(node_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm2_node_feats(node_feats)
		
		# Apply MLP for node features
		for layer in self.node_feats_MLP:
			node_feats = layer(node_feats)
		
		# Make second residual connection
		if self.residual:
			node_feats = node_feats_in2 + node_feats  # Make second node residual connection
		
		# Return node representations
		return node_feats
	
	def forward(self, g, node_feats, edge_feats):
		"""Perform a forward pass of a Graph Transformer to get final node representations."""
		node_feats = self.run_gt_layer(g, node_feats, edge_feats)
		return node_feats


class DGLGraphTransformer(nn.Module):
	"""A graph transformer, as a DGL module.
	"""
	def __init__(
			self,
			in_channels, 
			edge_features=10,
			num_hidden_channels=128,
            activ_fn=nn.SiLU(),
			transformer_residual=True,
			num_attention_heads=4,
			norm_to_apply='batch',
			dropout_rate=0.1,
			num_layers=4,
			**kwargs
			):
		"""Graph Transformer Layer
		
		Parameters
		----------
		in_channels : int
			Input channel size for nodes.
		edge_features : int
			Input channel size for edges.			
		num_hidden_channels : int
			Hidden channel size for both nodes and edges.
		activ_fn : Module
			Activation function to apply in MLPs.
		transformer_residual : bool
			Whether to use a transformer-residual update strategy for node features.
		num_attention_heads : int
			How many attention heads to apply to the input node features in parallel.
		norm_to_apply : str
			Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
		dropout_rate : float
			How much dropout (i.e. forget rate) to apply before activation functions.
		num_layers : int
			How many layers of geometric attention to apply.
		"""
		super(DGLGraphTransformer, self).__init__()
		
		# Initialize model parameters
		self.activ_fn = activ_fn
		self.transformer_residual = transformer_residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		# --------------------
		# Initializer Modules
		# --------------------
		# Define all modules related to edge and node initialization
		self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
		self.edge_encoder = nn.Linear(edge_features, num_hidden_channels) 
        # --------------------
		# Transformer Module
		# --------------------
		# Define all modules related to a variable number of Graph Transformer modules
		num_intermediate_layers = max(0, num_layers - 1)
		gt_block_modules = [GraphTransformerModule(
										num_hidden_channels=num_hidden_channels,
										activ_fn=activ_fn,
										residual=transformer_residual,
										num_attention_heads=num_attention_heads,
										norm_to_apply=norm_to_apply,
										dropout_rate=dropout_rate,
										num_layers=num_layers) for _ in range(num_intermediate_layers)]
		if num_layers > 0:
			gt_block_modules.extend([
							FinalGraphTransformerModule(
										num_hidden_channels=num_hidden_channels,
										activ_fn=activ_fn,
										residual=transformer_residual,
										num_attention_heads=num_attention_heads,
										norm_to_apply=norm_to_apply,
										dropout_rate=dropout_rate,
										num_layers=num_layers)])
		self.gt_block = nn.ModuleList(gt_block_modules)
	
	def forward(self, g, node_feats, edge_feats):		
		node_feats = self.node_encoder(node_feats)
		edge_feats = self.edge_encoder(edge_feats)
		
		g.ndata['x'] = node_feats
		g.edata['h'] = edge_feats	
		# Apply a given number of intermediate graph attention layers to the node and edge features given
		for gt_layer in self.gt_block[:-1]:
			node_feats, edge_feats = gt_layer(g, node_feats, edge_feats)
		
		# Apply final layer to update node representations by merging current node and edge representations
		node_feats = self.gt_block[-1](g, node_feats, edge_feats)
		return node_feats


def to_dense_batch_dgl(bg, feats, fill_value=0):
	max_num_nodes = int(bg.batch_num_nodes().max())
	#batch = feats.new_zeros(feats.size(0), dtype=torch.long)
	#batch = th.cat([th.full((1,int(x.cpu().numpy())), y) for x,y in zip(bg.batch_num_nodes(),range(bg.batch_size))],dim=1).reshape(-1).type(th.long)
	batch = th.cat([th.full((1,x.type(th.int)), y) for x,y in zip(bg.batch_num_nodes(),range(bg.batch_size))],dim=1).reshape(-1).type(th.long).to(bg.device)
	cum_nodes = th.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
	idx = th.arange(bg.num_nodes(), dtype=th.long, device=bg.device)
	idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
	size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
	out = feats.new_full(size, fill_value)
	out[idx] = feats
	out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
	
	mask = th.zeros(bg.batch_size * max_num_nodes, dtype=th.bool,
					device=bg.device)
	mask[idx] = 1
	mask = mask.view(bg.batch_size, max_num_nodes)  
	return out, mask

			
class RTMScore(nn.Module):
	def __init__(self, lig_model, prot_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15, 
					dist_threhold=1000):
		super(RTMScore, self).__init__()
		
		self.lig_model = lig_model
		self.prot_model = prot_model
		self.MLP = nn.Sequential(nn.Linear(in_channels*2, hidden_dim), 
								nn.BatchNorm1d(hidden_dim), 
								nn.ELU(), 
								nn.Dropout(p=dropout_rate)
								) 
		self.z_pi = nn.Linear(hidden_dim, n_gaussians)
		self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
		self.z_mu = nn.Linear(hidden_dim, n_gaussians)
		self.atom_types = nn.Linear(in_channels, 17)
		self.bond_types = nn.Linear(in_channels*2, 4)
		
		self.dist_threhold = dist_threhold	
    
	def forward(self, bgp, bgl):		
		h_l = self.lig_model(bgl, bgl.ndata['atom'].float(), bgl.edata['bond'].float())
		h_p = self.prot_model(bgp, bgp.ndata['feats'].float(), bgp.edata['feats'].float())
		
		h_l_x, l_mask = to_dense_batch_dgl(bgl, h_l, fill_value=0)
		h_p_x, p_mask = to_dense_batch_dgl(bgp, h_p, fill_value=0)

		h_l_pos, _ =  to_dense_batch_dgl(bgl, bgl.ndata["pos"], fill_value=0)
		h_p_pos, _ =  to_dense_batch_dgl(bgp, bgp.ndata["pos"], fill_value=0)
				
		(B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)
		self.B = B
		self.N_l = N_l
		self.N_p = N_p
		
		# Combine and mask
		h_l_x = h_l_x.unsqueeze(-2)
		h_l_x = h_l_x.repeat(1, 1, N_p, 1) # [B, N_l, N_t, C_out]
		
		h_p_x = h_p_x.unsqueeze(-3)
		h_p_x = h_p_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
		
		C = th.cat((h_l_x, h_p_x), -1)
		self.C_mask = C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
		self.C = C = C[C_mask]
		C = self.MLP(C)
		
		# Get batch indexes for ligand-target combined features
		C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
		C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]
			
		# Outputs
		pi = F.softmax(self.z_pi(C), -1)
		sigma = F.elu(self.z_sigma(C))+1.1
		mu = F.elu(self.z_mu(C))+1
		atom_types = self.atom_types(h_l)
		bond_types = self.bond_types(th.cat([h_l[bgl.edges()[0]],h_l[bgl.edges()[1]]], axis=1))
		
		dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3))[C_mask]
		return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch
	
	def compute_euclidean_distances_matrix(self, X, Y):
		# Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
		# (X-Y)^2 = X^2 + Y^2 -2XY
		X = X.double()
		Y = Y.double()
		
		dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2,    axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)	
		return th.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,24),10000).min(axis=-1)[0]

class Meter(object):
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []
		
        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None
		
    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration
		
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(th.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())
	
	
    def _finalize(self):
        """Prepare for evaluation.
	
        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.
	
        Returns
        -------
        mask : float32 tensor
            Binary mask indicating the existence of ground
            truth labels with shape (B, T), B for batch size
            and T for the number of tasks
        y_pred : float32 tensor
            Predicted labels with shape (B, T)
        y_true : float32 tensor
            Ground truth labels with shape (B, T)
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
	
        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean
		
        return mask, y_pred, y_true
	
    def _reduce_scores(self, scores, reduction='none'):
        """Finalize the scores to return.
		
        Parameters
        ----------
        scores : list of float
            Scores for all tasks.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
	
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))
	
    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate for multi-label prediction.
		
        Parameters
        ----------
        score_func : callable
            A score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score. The labels are in the form of 1D tensor.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
		
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)
	
    def pearson_r(self, reduction='none'):
        """Compute squared Pearson correlation coefficient.
		
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
		
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            #return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2
            return pearsonr(y_true.numpy(), y_pred.numpy())[0]
        return self.multilabel_score(score, reduction)
	
    def spearman_r(self, reduction='none'):
        """Compute squared Pearson correlation coefficient.
		
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
		
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return spearmanr(y_true.numpy(), y_pred.numpy())[0]
        return self.multilabel_score(score, reduction)	
    
    def mae(self, reduction='none'):
        """Compute mean absolute error.
		
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
		
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        return self.multilabel_score(score, reduction)
	
    def rmse(self, reduction='none'):
        """Compute root mean square error.
		
		Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
	
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return th.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
        return self.multilabel_score(score, reduction)
	
    def roc_auc_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.
	
        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.
	
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.
	
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), th.sigmoid(y_pred).numpy())
        return self.multilabel_score(score, reduction)
	
    def pr_auc_score(self, reduction='none'):
        """Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.
	
        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.
	
        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.
	
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), th.sigmoid(y_pred).numpy())
                return auc(recall, precision)
        return self.multilabel_score(score, reduction)
	
    def compute_metric(self, metric_name, reduction='none'):
        """Compute metric based on metric name.
	
        Parameters
        ----------
        metric_name : str
	
            * ``'r2'``: compute squared Pearson correlation coefficient
            * ``'mae'``: compute mean absolute error
            * ``'rmse'``: compute root mean square error
            * ``'roc_auc_score'``: compute roc-auc score
            * ``'pr_auc_score'``: compute pr-auc score
	
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks
	
        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if metric_name == 'rp':
            return self.pearson_r(reduction)
        elif metric_name == 'rs':
            return self.spearman_r(reduction)    
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        elif metric_name == 'return_pred_true':
            return self.return_pred_true()
        else:
            raise ValueError('Expect metric_name to be "rp" or "rs" or "mae" or "rmse" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))


class EarlyStopping(object):
    """Early stop tracker
	
    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.
	
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
	
    Examples
    --------
    Below gives a demo for a fake training process.
	
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping
	
    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)
	
    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break
	
    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            #dt = datetime.datetime.now()
            filename = 'early_stop.pth'
		
        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'
		
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
	
    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
	
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score
	
    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
	
        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
	
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
	
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        th.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)
	
    def load_checkpoint(self, model):
        '''Load the latest checkpoint
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(th.load(self.filename)['model_state_dict'])

def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
    normal = Normal(mu, sigma)
    #loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
    #loss = th.sum(loss * pi, dim=1)
    #loss = -th.log(loss)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    return loss



def run_a_train_epoch(epoch, model, data_loader, optimizer, aux_weight=0.001, device='cpu'):
	model.train()
	total_loss = 0
	mdn_loss = 0
	atom_loss = 0
	bond_loss = 0
	for batch_id, batch_data in enumerate(data_loader):
		pdbids, bgl, bgp = batch_data
		bgl = bgl.to(device)
		bgp = bgp.to(device)
		
		atom_labels = th.argmax(bgl.ndata["atom"][:,:17], dim=1, keepdim=False)
		bond_labels = th.argmax(bgl.edata["bond"][:,:4], dim=1, keepdim=False)
		
		pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgp, bgl)
		
		mdn = mdn_loss_fn(pi, sigma, mu, dist)
		mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
		mdn = mdn.mean()
		atom = F.cross_entropy(atom_types, atom_labels)
		bond = F.cross_entropy(bond_types, bond_labels)
		loss = mdn + (atom * aux_weight) + (bond * aux_weight)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		total_loss += loss.item() * bgl.batch_size
		mdn_loss += mdn.item() * bgl.batch_size
		atom_loss += atom.item() * bgl.batch_size
		bond_loss += bond.item() * bgl.batch_size
		
		if np.isinf(mdn_loss) or np.isnan(mdn_loss): break
		del bgl, bgp, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch, mdn, atom, bond, loss
		th.cuda.empty_cache()
		
	return total_loss / len(data_loader.dataset), mdn_loss / len(data_loader.dataset), atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)


def run_an_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False, dist_threhold=None, aux_weight=0.001, device='cpu'):
	model.eval()
	total_loss = 0
	mdn_loss = 0
	atom_loss = 0
	bond_loss = 0
	probs = []
	at_contrs = []
	res_contrs = []
	with th.no_grad():
		for batch_id, batch_data in enumerate(data_loader):
			pdbids, bgl, bgp = batch_data
			bgl = bgl.to(device)
			bgp = bgp.to(device)
			atom_labels = th.argmax(bgl.ndata["atom"][:,:17], dim=1, keepdim=False)
			bond_labels = th.argmax(bgl.edata["bond"][:,:4], dim=1, keepdim=False)
			
			pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgp, bgl)
			
			if pred or atom_contribution or res_contribution:
				prob = calculate_probablity(pi, sigma, mu, dist)
				if dist_threhold is not None:
					prob[th.where(dist > dist_threhold)[0]] = 0.
				
				batch = batch.to(device)
				if pred:
					probx = scatter_add(prob, batch, dim=0, dim_size=bgl.batch_size)
					probs.append(probx)
				if atom_contribution or res_contribution:				
					contribs = [prob[batch==i].reshape((bgl.batch_num_nodes()[i], bgp.batch_num_nodes()[i])) for i in range(bgl.batch_size)]
					if atom_contribution:
						at_contrs.extend([contribs[i].sum(1).cpu().detach().numpy() for i in range(bgl.batch_size)])
					if res_contribution:
						res_contrs.extend([contribs[i].sum(0).cpu().detach().numpy() for i in range(bgl.batch_size)])
			
			else:
				mdn = mdn_loss_fn(pi, sigma, mu, dist)
				mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
				mdn = mdn.mean()
				atom = F.cross_entropy(atom_types, atom_labels)
				bond = F.cross_entropy(bond_types, bond_labels)
				loss = mdn + (atom * aux_weight) + (bond * aux_weight)
				
				total_loss += loss.item() * bgl.batch_size
				mdn_loss += mdn.item() * bgl.batch_size
				atom_loss += atom.item() * bgl.batch_size
				bond_loss += bond.item() * bgl.batch_size
				
		
			del bgl, bgp, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch
			th.cuda.empty_cache()
	
	if atom_contribution or res_contribution:
		if pred:
			preds = th.cat(probs)
			return [preds.cpu().detach().numpy(),at_contrs,res_contrs]
		else:
			return [None, at_contrs,res_contrs]
	else:
		if pred:
			preds = th.cat(probs)
			return preds.cpu().detach().numpy()
		else:		
			return total_loss / len(data_loader.dataset), mdn_loss / len(data_loader.dataset), atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)


def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)
	
    return prob



def collate(data):
	pdbids, graphsl, graphsp = map(list, zip(*data))
	bgl = dgl.batch(graphsl)
	bgp = dgl.batch(graphsp)
	for nty in bgl.ntypes:
		bgl.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bgl.canonical_etypes:
		bgl.set_e_initializer(dgl.init.zero_initializer, etype=ety)
	for nty in bgp.ntypes:
		bgp.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bgp.canonical_etypes:
		bgp.set_e_initializer(dgl.init.zero_initializer, etype=ety)	
	return pdbids, bgl, bgp


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    #th.backends.cudnn.benchmark = False
    #th.backends.cudnn.deterministic = True
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

class PDBbindDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prots=None
				):
		if isinstance(ids,np.ndarray) or isinstance(ids,list):
			self.pdbids = ids
		else:
			try:
				self.pdbids = np.load(ids)
			except:
				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
		if isinstance(ligs,np.ndarray) or isinstance(ligs,tuple) or isinstance(ligs,list):
			if isinstance(ligs[0],dgl.DGLGraph):
				self.graphsl = ligs
			else:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
		else:
			try:
				self.graphsl, _ = load_graphs(ligs) 
			except:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
		
		if isinstance(prots,np.ndarray) or isinstance(prots,tuple) or isinstance(prots,list):
			if isinstance(prots[0],dgl.DGLGraph):
				self.graphsp = prots
			else:
				raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')	
		else:
			try:
				self.graphsp, _ = load_graphs(prots) 
			except:
				raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')
		
		self.graphsl = list(self.graphsl)
		self.graphsp = list(self.graphsp)
		assert len(self.pdbids) == len(self.graphsl) == len(self.graphsp)
		
	def __getitem__(self, idx): 
		""" Get graph and label by index
		
        Parameters
        ----------
        idx : int
            Item index
	
		Returns
		-------
		(dgl.DGLGraph, Tensor)
		"""
		return self.pdbids[idx], self.graphsl[idx], self.graphsp[idx]
	
	
	def __len__(self):
		"""Number of graphs in the dataset"""
		return len(self.pdbids)			
	
	
	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		#random.seed(seed)
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)),valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds
		
		
class VSDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prot=None,
				gen_pocket=False,
				cutoff=None,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		self.graphp=None
		self.graphsl=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		
		if isinstance(prot, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.prot = prot
			self.graphp = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(prot, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.graphp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					self.prot = pocket
					self.graphp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			
		if isinstance(ligs,np.ndarray) or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.graphsl = self._mol_to_graph()
			elif isinstance(ligs[0], dgl.DGLGraph):
				self.graphsl = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs.endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.graphsl = self._mol_to_graph()
			elif ligs.endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.graphsl = self._mol_to_graph()
			else:
				try:	
					self.graphsl,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
		
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.graphsl))]
		else:
			self.idsx = ids

		self.ids, self.graphsl = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.graphsl)))
		self.ids = list(self.ids)
		self.graphsl = list(self.graphsl)
		assert len(self.ids) == len(self.graphsl)
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)
		
	def __getitem__(self, idx): 
		""" Get graph and label by index
	
        Parameters
        ----------
        idx : int
            Item index
	
		Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
		return self.ids[idx], self.graphsl[idx], self.graphp
	
	def __len__(self):
		"""Number of graphs in the dataset"""
		return len(self.ids)	
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			print("failed to scoring for {} and {}".format(self.graphp, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None

METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
		"CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
		"TM","LU","HF","ZR","CE","U","PU","TH"] 
RES_MAX_NATOMS=24

def prot_to_graph(prot, cutoff):
	"""obtain the residue graphs"""
	u = mda.Universe(prot)
	g = dgl.DGLGraph()
	# Add nodes
	num_residues = len(u.residues)
	g.add_nodes(num_residues)
	
	res_feats = np.array([calc_res_features(res) for res in u.residues])
	g.ndata["feats"] = th.tensor(res_feats)
	edgeids, distm = obatin_edge(u, cutoff)	
	src_list, dst_list = zip(*edgeids)
	g.add_edges(src_list, dst_list)
	
	g.ndata["ca_pos"] = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))	
	g.ndata["center_pos"] = th.tensor(u.atoms.center_of_mass(compound='residues'))
	dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
	cadist = th.tensor([dis_matx_ca[i,j] for i,j in edgeids]) * 0.1
	dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
	cedist = th.tensor([dis_matx_center[i,j] for i,j in edgeids]) * 0.1
	edge_connect =  th.tensor(np.array([check_connect(u, x, y) for x,y in zip(src_list, dst_list)]))
	g.edata["feats"] = th.cat([edge_connect.view(-1,1), cadist.view(-1,1), cedist.view(-1,1), th.tensor(distm)], dim=1)
	g.ndata.pop("ca_pos")
	g.ndata.pop("center_pos")
	#res_max_natoms = max([len(res.atoms) for res in u.residues])
	g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
	#g.ndata["posmask"] = th.tensor([[1]* len(res.atoms)+[0]*(RES_MAX_NATOMS-len(res.atoms)) for res in u.residues]).bool()
	#g.ndata["atnum"] = th.tensor([len(res.atoms) for res in u.residues])
	return g


def obtain_ca_pos(res):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms("name CA").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)
		ca = xx.select_atoms("name CA")
		c = xx.select_atoms("name C")
		n = xx.select_atoms("name N")
		o = xx.select_atoms("name O")
		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
	except:
		return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
	try:
		if res.phi_selection() is not None:
			phi = res.phi_selection().dihedral.value()
		else:
			phi = 0
		if res.psi_selection() is not None:
			psi = res.psi_selection().dihedral.value()
		else:
			psi = 0
		if res.omega_selection() is not None:
			omega = res.omega_selection().dihedral.value()
		else:
			omega = 0
		if res.chi1_selection() is not None:
			chi1 = res.chi1_selection().dihedral.value()
		else:
			chi1 = 0
		return [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
	except:
		return [0, 0, 0, 0]

def calc_res_features(res):
	return np.array(one_of_k_encoding_unk(obtain_resname(res), 
										['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +          #32  residue type	
			obtain_self_dist(res) +  #5
			obtain_dihediral_angles(res) #4		
			)

def obtain_resname(res):
	if res.resname[:2] == "CA":
		resname = "CA"
	elif res.resname[:2] == "FE":
		resname = "FE"
	elif res.resname[:2] == "CU":
		resname = "CU"
	else:
		resname = res.resname.strip()
	
	if resname in METAL:
		return "M"
	else:
		return resname

##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
	edgeids = []
	dismin = []
	dismax = []
	for res1, res2 in permutations(u.residues, 2):
		dist = calc_dist(res1, res2)
		if dist.min() <= cutoff:
			edgeids.append([res1.ix, res2.ix])
			dismin.append(dist.min()*0.1)
			dismax.append(dist.max()*0.1)
	return edgeids, np.array([dismin, dismax]).T



def check_connect(u, i, j):
	if abs(i-j) != 1:
		return 0
	else:
		if i > j:
			i = j
		nb1 = len(u.residues[i].get_connections("bonds"))
		nb2 = len(u.residues[i+1].get_connections("bonds"))
		nb3 = len(u.residues[i:i+2].get_connections("bonds"))
		if nb1 + nb2 == nb3 + 1:
			return 1
		else:
			return 0
		
	

def calc_dist(res1, res2):
	#xx1 = res1.atoms.select_atoms('not name H*')
	#xx2 = res2.atoms.select_atoms('not name H*')
	#dist_array = distances.distance_array(xx1.positions,xx2.positions)
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array
	#return dist_array.max()*0.1, dist_array.min()*0.1



def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
       'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 
		'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 
		'Cu', 'Mn', 'Mo', 'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,'other']) + [atom.GetIsAromatic()]
                # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])	
    return np.array(results)


def calc_bond_features(bond, use_chirality=True):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


	
def load_mol(molpath, explicit_H=False, use_chirality=True):
	# load mol
	if re.search(r'.pdb$', molpath):
		mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
	elif re.search(r'.mol2$', molpath):
		mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
	elif re.search(r'.sdf$', molpath):			
		mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
	else:
		raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")	
	
	if use_chirality:
		Chem.AssignStereochemistryFrom3D(mol)
	return mol


def mol_to_graph(mol, explicit_H=False, use_chirality=True):
	"""
	mol: rdkit.Chem.rdchem.Mol
	explicit_H: whether to use explicit H
	use_chirality: whether to use chirality
	"""   	
				
	g = dgl.DGLGraph()
	# Add nodes
	num_atoms = mol.GetNumAtoms()
	g.add_nodes(num_atoms)
	
	atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
	if use_chirality:
		chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
		chiral_arr = np.zeros([num_atoms,3]) 
		for (i, rs) in chiralcenters:
			if rs == 'R':
				chiral_arr[i, 0] =1 
			elif rs == 'S':
				chiral_arr[i, 1] =1 
			else:
				chiral_arr[i, 2] =1 
		atom_feats = np.concatenate([atom_feats,chiral_arr],axis=1)
			
	g.ndata["atom"] = th.tensor(atom_feats)
	
	# obtain the positions of the atoms
	atomCoords = mol.GetConformer().GetPositions()
	g.ndata["pos"] = th.tensor(atomCoords)
	
	# Add edges
	src_list = []
	dst_list = []
	bond_feats_all = []
	num_bonds = mol.GetNumBonds()
	for i in range(num_bonds):
		bond = mol.GetBondWithIdx(i)
		u = bond.GetBeginAtomIdx()
		v = bond.GetEndAtomIdx()
		bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
		src_list.extend([u, v])
		dst_list.extend([v, u])		
		bond_feats_all.append(bond_feats)
		bond_feats_all.append(bond_feats)
	
	g.add_edges(src_list, dst_list)
	#normal_all = []
	#for i in etype_feature_all:
	#	normal = etype_feature_all.count(i)/len(etype_feature_all)
	#	normal = round(normal, 1)
	#	normal_all.append(normal)
	
	g.edata["bond"] = th.tensor(np.array(bond_feats_all))
	#g.edata["normal"] = th.tensor(normal_all)
	
	#dis_matx = distance_matrix(g.ndata["pos"], g.ndata["pos"])
	#g.edata["dist"] = th.tensor([dis_matx[i,j] for i,j in zip(*g.edges())]) * 0.1	
	return g



def mol_to_graph2(prot_path, lig_path, cutoff=10.0, explicit_H=False, use_chirality=True):
	prot = load_mol(prot_path, explicit_H=explicit_H, use_chirality=use_chirality) 
	lig = load_mol(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
	#gm = obtain_inter_graphs(prot, lig, cutoff=cutoff)
	#return gm
	#up = mda.Universe(prot)
	gp = prot_to_graph(prot, cutoff)
	gl = mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality)
	return gp, gl



def pdbbind_handle(pdbid, args):
	prot_path = "%s/%s/%s_prot/%s_p_pocket_%s.pdb"%(args.dir, pdbid, pdbid, pdbid, args.cutoff)
	lig_path = "%s/%s/%s_prot/%s_l.sdf"%(args.dir, pdbid, pdbid, pdbid)
	try: 
		gp, gl = mol_to_graph2(prot_path, 
							lig_path, 
							cutoff=args.cutoff,
							explicit_H=args.useH, 
							use_chirality=args.use_chirality)
	except:
		print("%s failed to generare the graph"%pdbid)
		gp, gl = None, None
		#gm = None
	return pdbid, gp, gl


def UserInput():
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('-d', '--dir', default=".",
						help='The directory to store the protein-ligand complexes.')	
	p.add_argument('-c', '--cutoff', default=None, type=float,
						help='the cutoff to determine the pocket')	
	p.add_argument('-o', '--outprefix', default="out",
						help='The output bin file.')	
	p.add_argument('-usH', '--useH', default=False, action="store_true",
						help='whether to use the explicit H atoms.')
	p.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
						help='whether to use chirality.')							
	p.add_argument('-p', '--parallel', default=False, action="store_true",
						help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')	
	
	args = p.parse_args()	
	return args



def main():
	args = UserInput()
	pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s"%(args.dir, x))]
	if args.parallel:
		results = Parallel(n_jobs=-1)(delayed(pdbbind_handle)(pdbid, args) for pdbid in pdbids)
	else:
		results = []
		for pdbid in pdbids:
			results.append(pdbbind_handle(pdbid, args))
	results = list(filter(lambda x: x[1] != None, results))
	#ids, graphs =  list(zip(*results))
	#np.save("%s_idsresx.npy"%args.outprefix, ids)
	#save_graphs("%s_plresx.bin"%args.outprefix, list(graphs))	
	ids, graphs_p, graphs_l =  list(zip(*results))
	np.save("%s_idsresz.npy"%args.outprefix, ids)
	save_graphs("%s_presz.bin"%args.outprefix, list(graphs_p))
	save_graphs("%s_lresz.bin"%args.outprefix, list(graphs_l))
	


if __name__ == '__main__':
	main()

