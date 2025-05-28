import torch
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import BatchNorm as PyGBatchNorm # PyG's BatchNorm for graph nodes
from torch_geometric.nn import LayerNorm as PyGLayerNorm # PyG's LayerNorm for graph nodes
from torch_geometric.utils import to_dense_batch # For Transformer processing

class GINELayerLight(nn.Module):
    def __init__(self, node_dim, # Input/Output dimension for node features in this layer
                 processed_edge_feat_dim, # Dimension of edge features GINEConv will use
                 dropout_rate=0.2):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(node_dim, 2 * node_dim), # MLP takes node_dim
            nn.BatchNorm1d(2 * node_dim),
            nn.ReLU(),
            nn.Linear(2 * node_dim, node_dim), # MLP outputs node_dim
        )
        self.conv = GINEConv(nn=mlp, eps=0.0, train_eps=True, edge_dim=processed_edge_feat_dim) # GINEConv uses processed_edge_feat_dim
        self.norm = PyGBatchNorm(node_dim) # BatchNorm for node_dim features
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr_processed):
        # No per-layer residual here; block-level residuals will be applied outside
        x_conv = self.conv(x, edge_index, edge_attr=edge_attr_processed)
        x = self.norm(x_conv)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class GINEClassifier(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_classes,
                 num_gnn_blocks=3,
                 num_gine_layers_per_block=2,
                 gnn_dropout=0.2, # Dropout rate within each GINELayer
                 use_dense_skip=True,
                 use_global_transformer=True,
                 num_transformer_layers=1, # Number of layers in nn.TransformerEncoder
                 num_transformer_heads=4,
                 transformer_ff_dim_factor=2, # Factor for transformer's feed-forward dim
                 transformer_dropout=0.1, # Dropout in TransformerEncoderLayer
                 classifier_dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dense_skip = use_dense_skip
        self.use_global_transformer = use_global_transformer

        # 1. Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge features will be encoded to a dimension suitable for GINEConv
        self.encoded_edge_dim_for_gine = hidden_dim # GINEConv can take edge_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * 2), # Inspired by EdgeCentricGNNImproved
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.encoded_edge_dim_for_gine)
        )

        # 2. GNN Blocks
        self.gnn_blocks_modulelist = nn.ModuleList()
        for _ in range(num_gnn_blocks):
            block_layers = nn.ModuleList()
            for _ in range(num_gine_layers_per_block):
                block_layers.append(
                    GINELayerLight(
                        node_dim=self.hidden_dim,
                        processed_edge_feat_dim=self.encoded_edge_dim_for_gine,
                        dropout_rate=gnn_dropout
                    )
                )
            self.gnn_blocks_modulelist.append(block_layers)
        
        if self.use_dense_skip:
            self.norm_for_dense_skip = PyGLayerNorm(hidden_dim)

        # 3. Global Transformer Interlayer (Optional)
        if self.use_global_transformer:
            self.transformer_input_norm = PyGLayerNorm(hidden_dim) # Normalize input to Transformer
            transformer_encoder_layer_impl = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_transformer_heads,
                dim_feedforward=hidden_dim * transformer_ff_dim_factor,
                dropout=transformer_dropout,
                activation='relu',
                batch_first=True # Our dense batch will be (batch_size, seq_len, features)
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer_impl,
                num_layers=num_transformer_layers
            )

        # 4. Readout and Classifier
        self.pool = global_mean_pool
        self.classifier_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Start with hidden_dim
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. Encoders
        x_encoded = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)

        # Store outputs for dense skip connection
        outputs_for_dense_skip = [x_encoded.clone()]

        # 2. GNN Blocks
        current_h = x_encoded
        for gnn_block in self.gnn_blocks_modulelist:
            h_block_input = current_h.clone() # For block-level residual
            for gine_layer_in_block in gnn_block:
                current_h = gine_layer_in_block(current_h, edge_index, edge_attr_encoded)
            
            current_h = current_h + h_block_input # Block-level residual connection
            outputs_for_dense_skip.append(current_h.clone())

        # Determine the features to pass to the next stage (Transformer or Pooling)
        if self.use_dense_skip:
            # Sum all collected outputs (initial encoded + each block's output)
            h_to_process_next = torch.zeros_like(x_encoded) # Initialize with correct shape and device
            for out_tensor in outputs_for_dense_skip:
                h_to_process_next = h_to_process_next + out_tensor
            h_to_process_next = self.norm_for_dense_skip(h_to_process_next)
        else:
            # Use only the output of the last GNN block
            h_to_process_next = current_h

        # 3. Global Transformer Interlayer (Optional)
        if self.use_global_transformer and hasattr(self, 'transformer_encoder'):
            # Convert to dense batch for Transformer: (total_nodes, dim) -> (batch_size, max_nodes, dim)
            dense_h, node_mask = to_dense_batch(h_to_process_next, batch, fill_value=0)
            # node_mask is True for real nodes, False for padding

            # Normalize before Transformer
            dense_h = self.transformer_input_norm(dense_h)

            # Transformer expects src_key_padding_mask where True means "ignore"
            padding_mask = ~node_mask
            transformer_out_dense = self.transformer_encoder(dense_h, src_key_padding_mask=padding_mask)
            
            # Convert back to sparse format by selecting only real node embeddings
            h_final_nodes = transformer_out_dense[node_mask]
        else:
            h_final_nodes = h_to_process_next

        # 4. Readout
        graph_embedding = self.pool(h_final_nodes, batch)

        # 5. Classifier
        out = self.classifier_fc(graph_embedding)
        return out

class GINELayer(nn.Module):
    """A single GINEConv layer followed by activation, dropout, and LayerNorm."""
    def __init__(self, in_dim, out_dim, edge_dim, dropout_rate=0.2, activation_fn=F.leaky_relu):
        super().__init__()
        # The nn for GINEConv: maps node features to the same dimension as output
        # If in_dim != out_dim, the GINEConv output will be out_dim due to its internal MLP.
        # The nn here typically transforms `x_j + edge_attr` or just `x_j`.
        # Let's assume the GINEConv's internal MLP handles the feature transformation size.
        # The nn inside GINEConv should map `in_dim` to `out_dim`.
        # Here, we define the MLP that GINEConv will use.
        # It processes concatenated node and edge features or just node features.
        # GINEConv nn should transform input features (hidden_dim) to output features (hidden_dim)
        nn_sequential = nn.Sequential(
            nn.Linear(in_dim, out_dim), # GINE process input dim to out dim
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.conv = GINEConv(nn_sequential, edge_dim=edge_dim, train_eps=True)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation_fn

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x, negative_slope=0.2) if self.activation == F.leaky_relu else self.activation(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x

class TransformerGNNLayer(nn.Module):
    """A single TransformerConv layer followed by activation, dropout, and LayerNorm."""
    def __init__(self, in_dim, out_dim_per_head, heads, edge_dim, dropout_rate=0.2, activation_fn=F.leaky_relu):
        super().__init__()
        self.conv = TransformerConv(in_dim, out_dim_per_head, heads=heads, concat=True,
                                    edge_dim=edge_dim, dropout=dropout_rate, bias=True) # TransformerConv has internal dropout
        self.norm = nn.LayerNorm(out_dim_per_head * heads) # Output dim is heads * out_dim_per_head
        self.dropout = nn.Dropout(dropout_rate) # Additional dropout after activation/norm
        self.activation = activation_fn

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x, negative_slope=0.2) if self.activation == F.leaky_relu else self.activation(x)
        # Dropout in TransformerConv is on attention weights. This dropout is on features.
        x = self.dropout(x)
        x = self.norm(x)
        return x

class EdgeCentricGNNImproved(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, edge_dim, num_gine_layers_per_block=3, dropout=0.2):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * 2), # hidden_dim*2 might be too large, let's keep original
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim) # Encoded edge_dim is hidden_dim
        )

        # --- GINE Blocks ---
        self.gine_block1_layers = nn.ModuleList()
        for _ in range(num_gine_layers_per_block):
            self.gine_block1_layers.append(GINELayer(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.bn_block1 = BatchNorm(hidden_dim) # PyG BatchNorm

        self.gine_block2_layers = nn.ModuleList()
        for _ in range(num_gine_layers_per_block):
            self.gine_block2_layers.append(GINELayer(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.bn_block2 = BatchNorm(hidden_dim)

        self.gine_block3_layers = nn.ModuleList()
        for _ in range(num_gine_layers_per_block):
            self.gine_block3_layers.append(GINELayer(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.bn_block3 = BatchNorm(hidden_dim)

        # --- Transformer Interlayers ---
        heads = 4
        transformer_out_per_head = hidden_dim // heads
        
        self.transformer_mid1 = TransformerGNNLayer(hidden_dim, transformer_out_per_head, heads, hidden_dim, dropout)
        self.transformer_mid2 = TransformerGNNLayer(hidden_dim, transformer_out_per_head, heads, hidden_dim, dropout)
        self.transformer_final = TransformerGNNLayer(hidden_dim, transformer_out_per_head, heads, hidden_dim, dropout)
        self.bn_final_transformer = BatchNorm(hidden_dim) # PyG BatchNorm after final transformer

        # --- Dense Skip Connection Normalization ---
        self.norm_dense_skip = nn.LayerNorm(hidden_dim) # For the sum of multiple skips

        # --- Readout and classification ---
        self.pool = global_mean_pool
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout), # Added dropout
            nn.Linear(hidden_dim * 2, hidden_dim), # Simplified slightly
            nn.ReLU(),
            nn.Dropout(dropout), # Added dropout
            nn.Linear(hidden_dim, output_dim),
        )
        # The original had many skips like self.skip_2 = nn.Linear(hidden_dim, hidden_dim)
        # These are effectively 1x1 convolutions. If the intention is a simple residual,
        # direct addition is fine. If it's a learnable transformation, they can be kept.
        # For optimization, let's assume direct addition for block residuals for now.

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. Encoders
        encoded_edge_attr = self.edge_encoder(edge_attr)
        x = self.node_encoder(x)

        # --- Block 1 (GINE Layers + Residual) ---
        x_res_block1 = x.clone()
        for layer in self.gine_block1_layers:
            x = layer(x, edge_index, encoded_edge_attr)
        x = x + x_res_block1 # Residual connection for the block
        x = self.bn_block1(x) # BatchNorm after block's residual sum
        x_after_block1 = x.clone() # For dense skip

        # --- Mid Layer 1 (TransformerConv) ---
        x = self.transformer_mid1(x, edge_index, encoded_edge_attr)
        
        # --- Block 2 (GINE Layers + Residual) ---
        x_res_block2 = x.clone()
        for layer in self.gine_block2_layers:
            x = layer(x, edge_index, encoded_edge_attr)
        x = x + x_res_block2
        x = self.bn_block2(x)
        x_after_block2 = x.clone() # For dense skip

        # --- Mid Layer 2 (TransformerConv) ---
        x = self.transformer_mid2(x, edge_index, encoded_edge_attr)

        # --- Block 3 (GINE Layers + Residual) ---
        x_res_block3 = x.clone()
        for layer in self.gine_block3_layers:
            x = layer(x, edge_index, encoded_edge_attr)
        x = x + x_res_block3
        x = self.bn_block3(x)

        # --- Accumulated Dense Skip Connections ---
        x = x + x_after_block1 + x_after_block2 # Summing up outputs from earlier major blocks
        x = self.norm_dense_skip(x) # Normalize the sum

        # --- Final Transformer Layer ---
        x = self.transformer_final(x, edge_index, encoded_edge_attr)
        x = self.bn_final_transformer(x) # Using PyG BatchNorm here
        
        # --- Readout and Classification ---
        x_pooled = self.pool(x, batch)
        out = self.fc(x_pooled)
        
        return out