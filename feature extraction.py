import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=14, out_channels=16, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(128 * 30 * 16)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=14, out_channels=24, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(1920 * 14 * 24)
        
        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=14, out_channels=28, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(960 * 14 * 28)
        
        # Dropout layer
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        x = x.view(-1, 14, 128 * 30)
        
        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)
        
        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)
        
        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, patch_size, input_size):
        super(VisionTransformer, self).__init__()

        assert input_size[0] % patch_size[0] == 0 and input_size[1] % patch_size[1] == 0, "Input size must be divisible by patch size"
        num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        patch_dim = hidden_dim // 2  # Split hidden dimension between patches and embeddings
        
        # Patch embeddings
        self.patch_embeddings = nn.Conv2d(1, patch_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2).transpose(1, 2)
        
        batch_size, num_patches, _ = patches.shape
        position_embeddings = self.position_embeddings[:, :num_patches, :]
        patches = patches + position_embeddings
        
        patches = patches.transpose(0, 1)
        output = self.transformer_encoder(patches)
        
        output = torch.mean(output, dim=0)
        output = self.linear(output)
        
        return output

class ConcatenatedModel(nn.Module):
    def __init__(self, num_idcnn_layers, idcnn_hidden_dim, idcnn_num_heads, idcnn_patch_size, idcnn_input_size,
                 vit_num_layers, vit_hidden_dim, vit_num_heads, vit_patch_size, vit_input_size):
        super(ConcatenatedModel, self).__init__()
        
        # IDCNN feature extractor
        self.idcnn = FeatureExtractor()
        
        # ViT feature extractor
        self.vit = VisionTransformer(vit_num_layers, vit_hidden_dim, vit_num_heads, vit_patch_size, vit_input_size)
        
        # Final linear layer after concatenation
        self.final_linear = nn.Linear(idcnn_hidden_dim + vit_hidden_dim, idcnn_hidden_dim + vit_hidden_dim)
        
    def forward(self, x_idcnn, x_vit):
        # Forward pass through IDCNN
        features_idcnn = self.idcnn(x_idcnn)
        
        # Forward pass through ViT
        features_vit = self.vit(x_vit)
        
        # Concatenate features from IDCNN and ViT
        combined_features = torch.cat((features_idcnn, features_vit), dim=1)
        
        # Apply final linear layer
        output = self.final_linear(combined_features)
        
        return output

if __name__ == '__main__':
    # Example usage to test the models
    idcnn_num_layers = 3
    idcnn_hidden_dim = 480
    idcnn_num_heads = 16
    idcnn_patch_size = (8, 1)  # Adjusted patch size for IDCNN (assuming)
    idcnn_input_size = (128, 30)

    vit_num_layers = 12
    vit_hidden_dim = 768
    vit_num_heads = 16
    vit_patch_size = (16, 16)
    vit_input_size = (227, 22)

    # Create an instance of the ConcatenatedModel
    concat_model = ConcatenatedModel(idcnn_num_layers, idcnn_hidden_dim, idcnn_num_heads, idcnn_patch_size, idcnn_input_size,
                                     vit_num_layers, vit_hidden_dim, vit_num_heads, vit_patch_size, vit_input_size)

    # Example usage (assuming input tensors x_idcnn and x_vit of appropriate shapes)
    # Replace with actual data from your dataset
    x_idcnn = torch.randn(10, 14, 128, 30)  # Example input tensor for IDCNN
    x_vit = torch.randn(10, 1, 227, 22)     # Example input tensor for ViT

    output = concat_model(x_idcnn, x_vit)
    print("Output shape:", output.shape)