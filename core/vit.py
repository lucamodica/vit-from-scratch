"""
This module contains the implementation of the Vision Transformer (ViT) model.
It includes the necessary classes and functions to build and train the ViT architecture.
"""

import torch as t
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
      super().__init__()
      
      self.conv2d = nn.Conv2d(
        in_channels=in_channels,
        out_channels=emb_size,
        kernel_size=patch_size,
        stride=patch_size,
        padding=0
      )
      
      self.flatten = nn.Flatten(
        start_dim=2, # H 
        end_dim=-1  # W
      )
      
    def forward(self, x: t.Tensor) -> t.Tensor:
      img_res = x.shape[-1] # assuming square images (height == width)
      assert img_res % self.conv2d.kernel_size[0] == 0, \
        f"Image resolution {img_res} must be divisible by patch size {self.conv2d.kernel_size[0]}"
      
      x = self.conv2d(x)  # (batch_size, emb_size, num_patches_height, num_patches_width)
      x = self.flatten(x) # (batch_size, emb_size, num_patches_height * num_patches_width)
      x = x.permute(0, 2, 1) # (batch_size, num_patches, emb_size)
      return x
        
class MultiheadSelfAttentionBlock(nn.Module):
  def __init__(
    self, 
    emb_size: int, 
    num_heads: int,
    attn_dropout: float = 0 # it seems they don't use dropout in msa  
  ):
    super().__init__()
    self.msa = nn.MultiheadAttention(
      embed_dim=emb_size,
      num_heads=num_heads,
      dropout=attn_dropout,
      batch_first=True # (batch_size, seq_len, emb_size)
    )
    self.ln = nn.LayerNorm(emb_size)
   
  def forward(self, x: t.Tensor) -> t.Tensor:
    x = self.ln(x)
    attn_out, _ = self.msa(x, x, x, need_weights=False)
    
    return attn_out
  
class MLPBlock(nn.Module):
  def __init__(self, emb_size: int, mlp_size: int, dropout: float = 0.1):
    super().__init__()
    
    self.ln = nn.LayerNorm(emb_size)
    
    self.mlp = nn.Sequential(
      nn.Linear(emb_size, mlp_size),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(mlp_size, emb_size),
      nn.Dropout(dropout),
    )
    
  def forward(self, x: t.Tensor) -> t.Tensor:
    x = self.ln(x)
    mlp_out = self.mlp(x)
    return mlp_out
  
class TransformerEncoderBlock(nn.Module):
  def __init__(
    self, 
    emb_size: int = 768, 
    num_heads: int = 12, 
    mlp_size: int = 3072, 
    attn_dropout: int = 0,
    mlp_dropout: int = 0.1
  ):
    super().__init__()
    
    self.msa_block = MultiheadSelfAttentionBlock(emb_size, num_heads, attn_dropout)
    self.mlp_block = MLPBlock(emb_size, mlp_size, mlp_dropout)
    
  def forward(self, x: t.Tensor) -> t.Tensor:
    out_attn = self.msa_block(x)
    out_attn = out_attn + x # skip-connection 1
    
    out_mlp = self.mlp_block(out_attn)
    out = out_mlp + out_attn # skip-connection 2
    
    return out
  
class ViT(nn.Module):
  def __init__(
    self,
    n_layers: int = 12,
    d_model: int = 768,
    in_channels: int = 3, # we assume those are RGB images
    patch_size: int = 16,
    img_size: int = 224, # from Table 3 of the paper
    nheads: int = 12,
    dim_mlp: int = 3072,
    emb_dropout: float = 0.1,
    attn_dropout: float = 0,
    mlp_dropout: float = 0.1,
    n_classes: int = 1000 # default from ImageNet
  ):
    super().__init__()
    # check the resolution of the image can be divided by the patch size
    assert img_size % patch_size == 0, \
        f"Image resolution {img_size} must be divisible by patch size {patch_size}"
    
    # patch embedding layer
    self.patch_embedding = PatchEmbedding(
      in_channels=in_channels,
      patch_size=patch_size,
      emb_size=d_model
    )
    
    # embedding layer for the class token
    self.class_embedding = nn.Parameter(data=t.randn((1, 1, d_model), requires_grad=True))
    
    # embedding layer for the position
    self.num_patches = img_size**2 // patch_size**2
    self.pos_embedding = nn.Parameter(
      data=t.randn((1, self.num_patches + 1, d_model), 
      requires_grad=True)
    )
    
    # embedding dropout
    self.emb_dropout = nn.Dropout(emb_dropout)
    
    # sequence of transformer encoder layers
    layers = []
    for _ in range(n_layers):
      layers.append(TransformerEncoderBlock(
        emb_size=d_model,
        num_heads=nheads,
        mlp_size=dim_mlp,
        attn_dropout=attn_dropout,
        mlp_dropout=mlp_dropout
      ))
    self.model = nn.Sequential(*layers)
    
    # classification head
    self.classif_head = nn.Sequential(
      nn.LayerNorm(normalized_shape=d_model),
      nn.Linear(in_features=d_model, out_features=n_classes)
    )
    
    
  def forward(self, x: t.Tensor) -> t.Tensor:
    batch_size = x.shape[0]
    
    # create the class token embedidng by expanding
    # it across the number of batch size
    class_token_emb = self.class_embedding.expand(batch_size, -1, -1)
    
    # create the patch embeddings
    x = self.patch_embedding(x)
    
    # concat patch and class token embedding
    x = t.cat([class_token_emb, x], dim=1)
        
    # add it to the patch embedding and applying Dropout
    # (according to the Section B.1 of the paper)
    x = x + self.pos_embedding
    x = self.emb_dropout(x)
    
    # applying the tranfomer enoder layers
    # to the patched input
    x = self.model(x)
    
    # passing the first element of the output
    # of the transfomer encoder layers to
    # the classifier head and returning the result
    out = self.classif_head(x[:, 0])
    
    return out
    
    
    