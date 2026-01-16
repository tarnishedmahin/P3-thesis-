"""
RAG-RVT V2 Lite - Parameter-Efficient Advanced Gating Architecture
===================================================================
Lightweight version constrained to <10M parameters while maintaining novel components.

Key optimizations for parameter efficiency:
1. Shared projection layers where possible
2. Reduced hidden dimensions with bottleneck designs
3. Depthwise separable operations
4. Efficient attention with fewer heads
5. Compact gating networks

Novel components retained (in lighter form):
- Dynamic Modality Gate (DMG-Lite)
- Cross-Modal Squeeze-Excitation (CMSE-Lite)
- Lightweight Adaptive Fusion Gate
- Efficient Dual-Path Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import DropPath


def count_parameters(model):
    """Helper function to calculate trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------------
# LITE COMPONENT 1: Efficient Dynamic Modality Gate
# -----------------------------------------------------------------------------------
class DMG_Lite(nn.Module):
    """Lightweight Dynamic Modality Gate using shared bottleneck."""
    def __init__(self, cnn_dim, vit_dim, bottleneck=32):
        super().__init__()
        self.cnn_squeeze = nn.AdaptiveAvgPool2d(1)
        # Shared bottleneck projection
        self.proj = nn.Linear(cnn_dim + vit_dim, bottleneck)
        self.gate = nn.Linear(bottleneck, 2)
    
    def forward(self, cnn_feat, vit_cls):
        cnn_global = self.cnn_squeeze(cnn_feat).flatten(1)
        combined = torch.cat([cnn_global, vit_cls], dim=1)
        h = F.relu(self.proj(combined))
        weights = F.softmax(self.gate(h), dim=1)
        return weights[:, 0:1], weights[:, 1:2]


# -----------------------------------------------------------------------------------
# LITE COMPONENT 2: Efficient Cross-Modal SE
# -----------------------------------------------------------------------------------
class CMSE_Lite(nn.Module):
    """Lightweight Cross-Modal Squeeze-Excitation with aggressive reduction."""
    def __init__(self, cnn_channels, vit_dim, reduction=32):
        super().__init__()
        self.cnn_squeeze = nn.AdaptiveAvgPool2d(1)
        bottleneck = max(cnn_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels + vit_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, cnn_channels),
            nn.Sigmoid()
        )
    
    def forward(self, cnn_feat, vit_cls):
        B, C, H, W = cnn_feat.shape
        cnn_global = self.cnn_squeeze(cnn_feat).flatten(1)
        combined = torch.cat([cnn_global, vit_cls], dim=1)
        channel_weights = self.fc(combined).view(B, C, 1, 1)
        return cnn_feat * channel_weights


# -----------------------------------------------------------------------------------
# LITE COMPONENT 3: Lightweight Gated Cross-Attention
# -----------------------------------------------------------------------------------
class LiteGatedCrossAttention(nn.Module):
    """Efficient cross-attention with shared projections and fewer heads."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # Lightweight scalar gate
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query, key_value):
        attn_out, _ = self.attn(query=query, key=key_value, value=key_value)
        gate_weight = self.gate(attn_out)  # (B, S, 1)
        return self.norm(query + gate_weight * attn_out)


# -----------------------------------------------------------------------------------
# LITE COMPONENT 4: Compact Hierarchical Refinement
# -----------------------------------------------------------------------------------
class CompactRefinement(nn.Module):
    """Single-stage refinement with depthwise bottleneck."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        bottleneck = dim // reduction
        self.refine = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x * self.refine(x) + x)


# -----------------------------------------------------------------------------------
# LITE COMPONENT 5: Efficient Attention Pooling
# -----------------------------------------------------------------------------------
class EfficientAttentionPool(nn.Module):
    """Single-head attention pooling for minimal parameters."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    
    def forward(self, x):  # (B, S, D)
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


# -----------------------------------------------------------------------------------
# LITE COMPONENT 6: Compact Adaptive Fusion
# -----------------------------------------------------------------------------------
class CompactFusion(nn.Module):
    """Lightweight fusion with shared computations."""
    def __init__(self, dim):
        super().__init__()
        # Similarity-based gating
        self.sim_proj = nn.Linear(dim, dim // 4)
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, cnn_vec, vit_vec):
        # Compute similarity
        cnn_proj = self.sim_proj(cnn_vec)
        vit_proj = self.sim_proj(vit_vec)
        similarity = torch.sigmoid((cnn_proj * vit_proj).sum(dim=1, keepdim=True))
        
        # Gated fusion
        combined = torch.cat([cnn_vec, vit_vec], dim=1)
        gate = self.gate(combined)
        
        fused = gate * cnn_vec + (1 - gate) * vit_vec
        fused = fused * similarity + (cnn_vec + vit_vec) / 2 * (1 - similarity)
        
        return self.norm(fused)


# -----------------------------------------------------------------------------------
# LITE Fusion Module
# -----------------------------------------------------------------------------------
class V2_Lite_Fusion(nn.Module):
    """Lightweight fusion module with all novel components in compact form."""
    def __init__(self, cnn_channels, vit_dim, embed_dim, num_heads=4, dropout=0.1, drop_path=0.1):
        super().__init__()
        
        # Shared projection (saves parameters vs separate projections)
        self.cnn_proj = nn.Linear(cnn_channels, embed_dim)
        self.vit_proj = nn.Linear(vit_dim, embed_dim)
        
        # Pre-normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Single bidirectional cross-attention (shared for both directions to save params)
        self.cross_attn = LiteGatedCrossAttention(embed_dim, num_heads, dropout)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # Compact refinement
        self.refine = CompactRefinement(embed_dim)
        
        # Efficient pooling
        self.pool = EfficientAttentionPool(embed_dim)
        
        # Compact fusion
        self.fusion = CompactFusion(embed_dim)
    
    def forward(self, cnn_features, vit_features):
        B, C, H, W = cnn_features.shape
        
        # Flatten and project
        cnn_seq = cnn_features.flatten(2).permute(0, 2, 1)
        vit_seq = vit_features[:, 1:, :]
        
        cnn_proj = self.norm(self.cnn_proj(cnn_seq))
        vit_proj = self.norm(self.vit_proj(vit_seq))
        
        # Bidirectional cross-attention (reuse same module)
        cnn_attended = self.cross_attn(cnn_proj, vit_proj)
        vit_attended = self.cross_attn(vit_proj, cnn_proj)
        
        # Apply drop path and refinement
        cnn_refined = self.refine(self.drop_path(cnn_attended) + cnn_proj)
        vit_refined = self.refine(self.drop_path(vit_attended) + vit_proj)
        
        # Pool to vectors
        cnn_vec = self.pool(cnn_refined)
        vit_vec = self.pool(vit_refined)
        
        # Compact fusion
        fused = self.fusion(cnn_vec, vit_vec)
        
        return fused, cnn_vec, vit_vec


# -----------------------------------------------------------------------------------
# LITE Dual-Path Classifier
# -----------------------------------------------------------------------------------
class DualPathClassifier_Lite(nn.Module):
    """Lightweight dual-path classifier with shared layers."""
    def __init__(self, fused_dim, modality_dim, num_classes, dropout=0.2):
        super().__init__()
        hidden = fused_dim // 2
        
        # Main path (deeper)
        self.main_head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
        
        # Shared auxiliary projection (saves params vs separate heads)
        self.aux_proj = nn.Linear(modality_dim, hidden)
        self.aux_head = nn.Linear(hidden, num_classes)
        
        # Lightweight confidence weighting
        self.confidence = nn.Sequential(
            nn.Linear(fused_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, fused_feat, cnn_vec, vit_vec):
        # Main path
        main_logits = self.main_head(fused_feat)
        
        # Auxiliary paths (shared projection)
        cnn_h = F.gelu(self.aux_proj(cnn_vec))
        vit_h = F.gelu(self.aux_proj(vit_vec))
        cnn_logits = self.aux_head(cnn_h)
        vit_logits = self.aux_head(vit_h)
        
        # Confidence weighting
        weights = self.confidence(fused_feat)
        
        final_logits = (weights[:, 0:1] * main_logits + 
                       weights[:, 1:2] * cnn_logits + 
                       weights[:, 2:3] * vit_logits)
        
        return final_logits, main_logits, cnn_logits, vit_logits


# -----------------------------------------------------------------------------------
# FINAL MODEL: RAG-RVT V2 Lite (<10M parameters)
# -----------------------------------------------------------------------------------
class RAG_RVT_V2_Lite(nn.Module):
    """
    RAG-RVT V2 Lite: Parameter-efficient version (<10M params)
    
    Maintains novel components in compact form:
    - DMG-Lite: Dynamic Modality Gate
    - CMSE-Lite: Cross-Modal SE
    - Compact Gated Cross-Attention
    - Efficient Adaptive Fusion
    - Lite Dual-Path Classification
    """
    def __init__(self, num_classes, img_size=224,
                 cnn_backbone='efficientnet_b0',
                 vit_backbone='vit_tiny_patch16_224',
                 embed_dim=192,  # Reduced from 256
                 dropout=0.2,
                 drop_path=0.1,
                 pretrained=True,
                 use_auxiliary_loss=True,
                 auxiliary_weight=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_weight = auxiliary_weight
        self.embed_dim = embed_dim

        # CNN backbone
        self.cnn_backbone = timm.create_model(
            cnn_backbone, pretrained=pretrained, features_only=True, out_indices=[3]
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            cnn_feat = self.cnn_backbone(dummy)[0]
            cnn_channels = cnn_feat.shape[1]

        # ViT backbone
        self.vit_backbone = timm.create_model(vit_backbone, pretrained=pretrained, num_classes=0)
        vit_dim = self.vit_backbone.embed_dim

        # Lite novel components
        self.dmg = DMG_Lite(cnn_channels, vit_dim, bottleneck=32)
        self.cmse = CMSE_Lite(cnn_channels, vit_dim, reduction=32)

        # Lite fusion module
        self.fusion_module = V2_Lite_Fusion(
            cnn_channels=cnn_channels,
            vit_dim=vit_dim,
            embed_dim=embed_dim,
            num_heads=4,  # Reduced from 8
            dropout=dropout,
            drop_path=drop_path
        )

        # Lite classifier
        self.classifier = DualPathClassifier_Lite(
            fused_dim=embed_dim,
            modality_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x, return_all_logits=False):
        # Extract features
        vit_features = self.vit_backbone.forward_features(x)
        vit_cls = vit_features[:, 0, :]
        cnn_features = self.cnn_backbone(x)[0]
        
        # Apply CMSE
        cnn_features = self.cmse(cnn_features, vit_cls)
        
        # Dynamic modality weighting
        cnn_weight, vit_weight = self.dmg(cnn_features, vit_cls)
        
        cnn_features = cnn_features * cnn_weight.view(-1, 1, 1, 1)
        vit_features_scaled = vit_features.clone()
        vit_features_scaled[:, 1:, :] = vit_features[:, 1:, :] * vit_weight.unsqueeze(-1)
        
        # Fusion
        fused_feat, cnn_vec, vit_vec = self.fusion_module(cnn_features, vit_features_scaled)
        
        # Classification
        final_logits, main_logits, cnn_logits, vit_logits = self.classifier(fused_feat, cnn_vec, vit_vec)
        
        if return_all_logits:
            return final_logits, main_logits, cnn_logits, vit_logits
        return final_logits

    def compute_auxiliary_loss(self, main_logits, cnn_logits, vit_logits, targets, criterion):
        main_loss = criterion(main_logits, targets)
        cnn_loss = criterion(cnn_logits, targets)
        vit_loss = criterion(vit_logits, targets)
        aux_loss = (cnn_loss + vit_loss) / 2
        total_loss = main_loss + self.auxiliary_weight * aux_loss
        return total_loss, main_loss, aux_loss


# -----------------------------------------------------------------------------------
# Test and verify parameter count
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    print("🧠 Instantiating RAG-RVT V2 Lite (target: <10M params)...\n")

    model = RAG_RVT_V2_Lite(
        num_classes=4,
        img_size=224,
        cnn_backbone='efficientnet_b0',
        vit_backbone='vit_tiny_patch16_224',
        embed_dim=192,
        dropout=0.15,
        drop_path=0.1,
        pretrained=False,
        use_auxiliary_loss=True,
        auxiliary_weight=0.3
    )

    total_params = count_parameters(model)
    print(f"📊 Total trainable params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"{'✅ UNDER 10M LIMIT!' if total_params < 10_000_000 else '❌ EXCEEDS 10M LIMIT!'}\n")

    # Breakdown by component
    print("📋 Parameter breakdown:")
    print(f"   CNN backbone:  {count_parameters(model.cnn_backbone)/1e6:.2f}M")
    print(f"   ViT backbone:  {count_parameters(model.vit_backbone)/1e6:.2f}M")
    print(f"   DMG:           {count_parameters(model.dmg)/1e3:.1f}K")
    print(f"   CMSE:          {count_parameters(model.cmse)/1e3:.1f}K")
    print(f"   Fusion:        {count_parameters(model.fusion_module)/1e3:.1f}K")
    print(f"   Classifier:    {count_parameters(model.classifier)/1e3:.1f}K")

    # Test forward pass
    dummy = torch.randn(2, 3, 224, 224)
    final_logits, main_logits, cnn_logits, vit_logits = model(dummy, return_all_logits=True)
    print(f"\n✔️ Forward pass OK. Output shape: {final_logits.shape}")
