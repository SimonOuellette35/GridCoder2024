import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :] + x
        return self.dropout(x)

class LVM(nn.Module):

    def __init__(self, num_input_channels, num_classes, emb_dim, max_seq_length):
        super(LVM, self).__init__()

        self.emb_dim = emb_dim
        self.max_seq_length = max_seq_length

        # Increase number of channels to double parameters
        self.input_conv = self._make_conv_block(num_input_channels, 128)  # Increased channels
        self.down1 = self._make_conv_block(128, 256)
        self.down2 = self._make_conv_block(256, 512)
        self.down3 = self._make_conv_block(512, 1024)
        #self.down4 = self._make_conv_block(1024, 2048)  # Increased channels

        self.bridge = self._make_conv_block(2048, 2048)  # Increased channels

        #self.up4 = self._make_conv_block(6144, 2048)  # Increased channels
        self.up3 = self._make_conv_block(3072, 1024)
        self.up2 = self._make_conv_block(1536, 512)
        self.up1 = self._make_conv_block(768, 512)  # Increased channels

        self.final_conv = nn.Conv2d(512, 512, kernel_size=1)

        # Add TransformerDecoder layer
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_dim, nhead=4, dropout=0.),
            num_layers=1
        )
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(d_model=emb_dim, max_len=max_seq_length, dropout=0.)
        self.target_embedding = nn.Embedding(num_classes, emb_dim)
        
        # Add a linear layer to project the flattened CNN output to the transformer dimension
        self.cnn_projection = nn.Linear(512 * 30 * 30, emb_dim)
        
        # Add final output layer
        self.output_layer = nn.Linear(emb_dim, num_classes)
        
        self.relu = nn.ReLU(inplace=True)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward_single(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        return x1, x2, x3, x4#, x5

    def forward(self, x1, x2):
        # Process each input separately
        feat1_1, feat1_2, feat1_3, feat1_4 = self.forward_single(x1)
        feat2_1, feat2_2, feat2_3, feat2_4 = self.forward_single(x2)

        # Combine the deepest features
        combined = torch.cat((feat1_4, feat2_4), dim=1)

        # Bridge
        bridge = self.bridge(combined)

        # Upsampling path
        up3 = self.up3(torch.cat((bridge, feat1_3, feat2_3), dim=1))
        up2 = self.up2(torch.cat((up3, feat1_2, feat2_2), dim=1))
        up1 = self.up1(torch.cat((up2, feat1_1, feat2_1), dim=1))

        output = self.final_conv(up1)

        # Flatten and project CNN output to transformer dimension
        flattened_output = torch.reshape(output, (output.size(0), -1))
        projected_output = self.cnn_projection(flattened_output)
        
        # Reshape projected_output to be 3D: (seq_len, batch_size, emb_dim)
        projected_output = projected_output.unsqueeze(0)
        
        # Initialize the target sequence with the special token 3
        target_seq = torch.full((x1.size(0), 1), 3, dtype=torch.long, device=x1.device)
        
        # Initialize list to store logits for each step
        all_logits = []

        for _ in range(self.max_seq_length):
            # Get target embeddings and apply positional encoding
            target_embeddings = self.target_embedding(target_seq).permute(1, 0, 2)
            target_embeddings = self.pos_encoder(target_embeddings)

            # Pass through transformer decoder
            transformer_output = self.transformer_decoder(target_embeddings, projected_output)

            # Generate logits for the last token
            logits = self.output_layer(transformer_output[-1])
            all_logits.append(logits)

            # Get the predicted token
            predicted_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the predicted token to the target sequence
            target_seq = torch.cat([target_seq, predicted_token], dim=1)

        # Stack all logits
        logits = torch.stack(all_logits, dim=1)
        return logits
    
    def predict(self, x1, x2, target_seq):
        # Get the encoder output
        feat1_1, feat1_2, feat1_3, feat1_4 = self.forward_single(x1)
        feat2_1, feat2_2, feat2_3, feat2_4 = self.forward_single(x2)

        # Combine the deepest features
        combined = torch.cat((feat1_4, feat2_4), dim=1)

        # Bridge
        bridge = self.bridge(combined)

        # Upsampling path
        up3 = self.up3(torch.cat((bridge, feat1_3, feat2_3), dim=1))
        up2 = self.up2(torch.cat((up3, feat1_2, feat2_2), dim=1))
        up1 = self.up1(torch.cat((up2, feat1_1, feat2_1), dim=1))

        output = self.final_conv(up1)

        # Flatten and project CNN output to transformer dimension
        flattened_output = torch.reshape(output, (output.size(0), -1))
        projected_output = self.cnn_projection(flattened_output)
        
        # Apply positional encoding to the encoder (CNN) output
        projected_output = projected_output.unsqueeze(0)
        
        # Get target embeddings
        target_embeddings = self.target_embedding(target_seq).permute(1, 0, 2)
        target_embeddings = self.pos_encoder(target_embeddings[:self.max_seq_length])

        # Pass through transformer decoder
        transformer_output = self.transformer_decoder(target_embeddings, projected_output)

        # Generate logits for the last token
        logits = self.output_layer(transformer_output[-1])

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1)

        return probabilities


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

