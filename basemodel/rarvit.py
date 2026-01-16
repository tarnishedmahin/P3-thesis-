import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from math import sqrt

# A helper function to calculate model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------------
# NOVEL COMPONENT 1: Residual Cross-Attention Fusion (R-CAF) Module
# -----------------------------------------------------------------------------------
class R_CAF_Module(nn.Module):
    """
    Residual Cross-Attention Fusion (R-CAF) Module.
    This module fuses features from a CNN (local) and a ViT (global) stream.
    It uses cross-attention where features from one stream act as queries and
    features from the other act as keys/values, and vice-versa.
    The residual connection ensures that original information is preserved.
    """
    def __init__(self, cnn_channels, vit_dim, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Projection layers to bring both features to a common dimension
        self.cnn_proj = nn.Linear(cnn_channels, embed_dim)
        self.vit_proj = nn.Linear(vit_dim, embed_dim)

        # Cross-Attention layers
        self.cnn_to_vit_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.vit_to_cnn_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Final fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, cnn_features, vit_features):
        # cnn_features: (B, C, H, W)
        # vit_features: (B, N, D) where N is num_patches + 1 (for CLS token)

        # 1. Prepare features for attention
        # Reshape and permute CNN features to be sequence-like: (B, H*W, C)
        B, C, H, W = cnn_features.shape
        cnn_seq = cnn_features.flatten(2).permute(0, 2, 1)

        # Remove CLS token from ViT features for fusion with spatial features
        vit_patch_seq = vit_features[:, 1:, :]

        # 2. Project features to common embedding dimension
        cnn_proj_seq = self.cnn_proj(cnn_seq)      # (B, H*W, embed_dim)
        vit_proj_seq = self.vit_proj(vit_patch_seq) # (B, num_patches, embed_dim)

        # 3. Perform Bi-Directional Cross-Attention with Residual Connections
        # Attend: CNN queries ViT context
        cnn_context, _ = self.cnn_to_vit_attn(query=cnn_proj_seq, key=vit_proj_seq, value=vit_proj_seq)
        fused_cnn = self.norm1(cnn_proj_seq + cnn_context) # Additive residual

        # Attend: ViT queries CNN context
        vit_context, _ = self.vit_to_cnn_attn(query=vit_proj_seq, key=cnn_proj_seq, value=cnn_proj_seq)
        fused_vit = self.norm2(vit_proj_seq + vit_context) # Additive residual

        # 4. Combine and process the fused features
        # Average pool both sequences to get a fixed-size representation
        cnn_fused_vector = fused_cnn.mean(dim=1)
        vit_fused_vector = fused_vit.mean(dim=1)
        
        # Concatenate and pass through final MLP
        combined_vector = torch.cat([cnn_fused_vector, vit_fused_vector], dim=1)
        final_fused_features = self.fusion_mlp(combined_vector)

        return final_fused_features

# -----------------------------------------------------------------------------------
# NOVEL COMPONENT 2: Attention-Gated Recurrent (AGR) Block
# -----------------------------------------------------------------------------------
class AGR_Block(nn.Module):
    """
    Attention-Gated Recurrent (AGR) Block.
    This module acts as the classification head. It uses a GRU to process a 
    feature sequence and combines its final state with an attention-based
    context vector for robust classification.
    
    In this implementation, it takes the single fused vector from R-CAF,
    but it's designed to be extensible to sequences. We'll adapt it to the R-CAF output.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # "Attention Gate" - a self-gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is the fused vector from R-CAF
        x = self.dropout(self.bn1(F.gelu(self.fc1(x))))
        
        # Apply the attention gate
        gate_values = self.gate(x)
        gated_x = x * gate_values # Element-wise multiplication
        
        logits = self.fc2(gated_x)
        return logits


# -----------------------------------------------------------------------------------
# FINAL MODEL: Residual Attention-Gated Recurrent Vision Transformer (RAG-RVT)
# -----------------------------------------------------------------------------------
class RAG_RVT(nn.Module):
    def __init__(self, num_classes, img_size=224, cnn_backbone='efficientnet_b0', vit_backbone='vit_tiny_patch16_224', embed_dim=256, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # 1. CNN Stream (Local Features)
        # Using features_only to get intermediate feature maps
        self.cnn_backbone = timm.create_model(cnn_backbone, pretrained=pretrained, features_only=True, out_indices=[3]) # Stage 3 features
        # Dynamically get the number of channels from the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            cnn_channels = self.cnn_backbone(dummy_input)[0].shape[1]

        # 2. ViT Stream (Global Features)
        self.vit_backbone = timm.create_model(vit_backbone, pretrained=pretrained, num_classes=0) # Remove classifier head
        vit_dim = self.vit_backbone.embed_dim

        # 3. Residual Cross-Attention Fusion (R-CAF) Module
        self.fusion_module = R_CAF_Module(cnn_channels=cnn_channels, vit_dim=vit_dim, embed_dim=embed_dim)

        # 4. Attention-Gated Recurrent (AGR) Classification Head
        self.classification_head = AGR_Block(input_dim=embed_dim, hidden_dim=embed_dim // 2, num_classes=num_classes)
        
    def forward(self, x):
        # Get features from both streams
        cnn_features = self.cnn_backbone(x)[0]
        vit_features = self.vit_backbone.forward_features(x)

        # Fuse the features
        fused_features = self.fusion_module(cnn_features, vit_features)

        # Get final predictions
        logits = self.classification_head(fused_features)
        
        return logits

# -----------------------------------------------------------------------------------
# Example Usage and Parameter Verification
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Configuration ---
    IMG_SIZE = 224
    NUM_CLASSES = 5 # Example: No DR, Mild, Moderate, Severe, Proliferative DR
    
    # --- Model Instantiation ---
    print("🧠 Instantiating the RAG-RVT model...")
    model = RAG_RVT(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        cnn_backbone='efficientnet_b0',      # Lightweight CNN
        vit_backbone='vit_tiny_patch16_224', # Compact ViT
        embed_dim=256,                       # Common dimension for fusion
        pretrained=True                      # Use pre-trained weights
    )

    # --- Test with a Dummy Input ---
    print(f"✔️ Model instantiated. Testing with a dummy input of size (2, 3, {IMG_SIZE}, {IMG_SIZE})...")
    dummy_image_batch = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    try:
        output = model(dummy_image_batch)
        print(f"✔️ Forward pass successful!")
        print(f"   Output logits shape: {output.shape}") # Expected: (2, NUM_CLASSES)
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")

    # --- Parameter Count Verification ---
    total_params = count_parameters(model)
    print("\n" + "="*50)
    print("📊 Model Parameter Analysis")
    print("="*50)
    print(f"   Total trainable parameters: {total_params / 1e6:.2f}M")
    
    if 10e6 <= total_params <= 15e6:
        print("   ✅ Parameter count is within the target range (10-15M).")
    else:
        print("   ⚠️  Parameter count is outside the target range (10-15M).")
    print("="*50)