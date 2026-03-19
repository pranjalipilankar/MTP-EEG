import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin


#####################################################################################################
#                                    BDE Projection Layer Class                                     #
#####################################################################################################
class BDEProjectionLayer(nn.Module):
    """
    Projects Band Differential Entropy (BDE) tokens into a higher-dimensional embedding space.

    Args:
        bde_dim (int): Dimension of BDE features.
        embed_dim (int): Output embedding dimension.
    """

    def __init__(self, bde_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(bde_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Shape (batch_size, num_electrodes, bde_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        return self.linear(x)


#####################################################################################################
#                             Electrode Identity Embedding Class                                    #
#####################################################################################################
class ElectrodeIdentityEmbedding(nn.Module):
    """
    Adds a learnable identity embedding to each electrode position via element-wise addition.

    Args:
        num_electrodes (int): Number of EEG electrodes.
        embed_dim (int): Output embedding dimension.
    """

    def __init__(self, num_electrodes, embed_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, num_electrodes, embed_dim))

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        return x + self.embedding


#####################################################################################################
#                                       PreNorm Class                                               #
#####################################################################################################
class PreNorm(nn.Module):
    """
    Applies LayerNorm before the given function.

    Args:
        embed_dim (int): Output embedding dimension.
        fn (nn.Module): Model layer to call after normalization.
    """

    def __init__(self, embed_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        return self.fn(self.norm(x), **kwargs)


#####################################################################################################
#                                     Feed-Forward Class                                            #
#####################################################################################################
class FeedForward(nn.Module):
    """
    Feed-forward network subpart of the InterCortical Attention Block.

    Args:
        embed_dim (int): Output embedding dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        return self.net(x)


#####################################################################################################
#                                  Inter Cortical MHSA Class                                        #
#####################################################################################################
class InterCorticalMHSA(nn.Module):
    """
    Multi-head self-attention module for capturing inter-cortical neural dependencies.

    Args:
        embed_dim (int): Output embedding dimension.
        heads (int): Number of attention heads.
        head_dim (int): Embedding dimension per head.
        dropout (float): Dropout probability.
    """

    def __init__(self, embed_dim, heads=4, head_dim=16, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = head_dim**-0.5

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        b, n, _ = x.shape
        qkv = self.to_qkv(x)  # shape: (b, n, 3 * inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


#####################################################################################################
#                          Inter-Cortical Attention Block Class                                     #
#####################################################################################################
class InterCorticalAttentionBlock(nn.Module):
    """
    Transformer block composed of inter-cortical multi-headed self-attention and feed-forward network.

    Args:
        embed_dim (int): Output embedding dimension.
        heads (int): Number of attention heads.
        head_dim (int): Embedding dimension per head.
        mlp_hidden_dim (int): Hidden layer dimension in the feed-forward network.
        dropout (float): Dropout probability.
    """

    def __init__(self, embed_dim, heads, head_dim, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = PreNorm(
            embed_dim, InterCorticalMHSA(embed_dim, heads, head_dim, dropout)
        )
        self.feed_forward = PreNorm(
            embed_dim, FeedForward(embed_dim, mlp_hidden_dim, dropout)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_electrodes, embed_dim)
        """
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x


#####################################################################################################
#                                  Classification-Head Class                                        #
#####################################################################################################
class ClassificationHead(nn.Module):
    """
    Final classification layer for affective state prediction.

    Args:
        embed_dim (int): Output embedding dimension.
        num_classes (int): Number of affective classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, embed_dim, num_classes, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_electrodes, embed_dim)

        Returns:
            Tensor: Shape (batch_size, num_classes)
        """
        x = x.mean(dim=1)  # global average pooling over electrodes
        x = self.norm(x)
        return self.mlp(x)


#####################################################################################################
#                                          RBTransformer                                            #
#####################################################################################################
class RBTransformer(nn.Module, PyTorchModelHubMixin):
    """
    Final end-to-end RBTransformer model for EEG-based affective state classification.

    Args:
        num_electrodes (int): Number of EEG electrodes.
        bde_dim (int): Dimension of BDE features.
        embed_dim (int): Output embedding dimension.
        depth (int): Number of stacked InterCorticalAttentionBlock layers.
        heads (int): Number of attention heads per block.
        head_dim (int): Embedding dimension per attention head.
        mlp_hidden_dim (int): Hidden layer size in feed-forward networks.
        dropout (float): Dropout probability.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        num_electrodes=14,
        bde_dim=4,
        embed_dim=64,
        depth=3,
        heads=4,
        head_dim=16,
        mlp_hidden_dim=128,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        # Configure and initialize model layers of RBTransformer
        self.bde_proj_layer = BDEProjectionLayer(bde_dim, embed_dim)
        self.electrode_id_embedding = ElectrodeIdentityEmbedding(
            num_electrodes, embed_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.intercortical_attention_blocks = nn.ModuleList(
            [
                InterCorticalAttentionBlock(
                    embed_dim, heads, head_dim, mlp_hidden_dim, dropout
                )
                for _ in range(depth)
            ]
        )
        self.classification_head = ClassificationHead(embed_dim, num_classes, dropout)

        # Save model configuration for reproducibility
        self.config = {
            "num_electrodes": num_electrodes,
            "bde_dim": bde_dim,
            "embed_dim": embed_dim,
            "depth": depth,
            "heads": heads,
            "head_dim": head_dim,
            "mlp_hidden_dim": mlp_hidden_dim,
            "dropout": dropout,
            "num_classes": num_classes,
        }

    def forward(self, x):
        """
        Forward pass of the RBTransformer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_electrodes, bde_dim)

        Returns:
            Tensor: Logits of shape (batch_size, num_classes)
        """
        # Flatten extra dimension if present (e.g., from dataloader batch shapes)
        if x.dim() > 3:
            x = x.view(x.size(0), -1, self.config["bde_dim"])

        # Project BDE features into embedding space
        x = self.bde_proj_layer(x)

        # Add learnable electrode identity embeddings
        x = self.electrode_id_embedding(x)

        # Apply dropout post-embedding
        x = self.dropout(x)

        # Forward through stacked InterCorticalAttention blocks
        for block in self.intercortical_attention_blocks:
            x = block(x)

        # Final classification using deep MLP head
        return self.classification_head(x)


if __name__ == "__main__":
    dataset_config = {
        "deap": {
            "num_electrodes": 32,
            "valence": {"binary": 2, "multiclass": 9},
            "arousal": {"binary": 2, "multiclass": 9},
            "dominance": {"binary": 2, "multiclass": 9},
        },
        "dreamer": {
            "num_electrodes": 14,
            "valence": {"binary": 2, "multiclass": 5},
            "arousal": {"binary": 2, "multiclass": 5},
            "dominance": {"binary": 2, "multiclass": 5},
        },
        "seed": {"num_electrodes": 62, "emotion": {"multiclass": 3}},
    }

    dataset_name = "seed"
    label_name = "emotion"
    task_type = "multiclass"
    bde_dim = 4
    batch_size = 16

    num_electrodes = dataset_config[dataset_name]["num_electrodes"]
    num_classes = dataset_config[dataset_name][label_name][task_type]

    dummy_input = torch.randn(batch_size, num_electrodes, bde_dim)

    model = RBTransformer(
        num_electrodes=num_electrodes,
        bde_dim=bde_dim,
        embed_dim=64,
        depth=4,
        heads=6,
        head_dim=16,
        mlp_hidden_dim=128,
        dropout=0.1,
        num_classes=num_classes,
    )

    logits = model(dummy_input)
    print(
        f"Dataset: {dataset_name.upper()} | Label: {label_name} | Task: {task_type} | Logits shape: {logits.shape}"
    )