<<<<<<< HEAD
"""Seq2Seq model with bidirectional LSTM encoder-decoder, layer normalization and multi-head attention."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LayerNorm(nn.Module):
    """Layer Normalization module"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, seq_len, features] or [batch_size, features]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Bidirectional LSTM Encoder with layer normalization

        Args:
            input_dim: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Size of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embed_norm = LayerNorm(embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)

        self.output_norm = LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths=None):
        """
        Args:
            src: Source tensor [batch_size, src_len]
            lengths: Optional unpadded source lengths [batch_size]

        Returns:
            outputs: Output features [batch_size, src_len, hidden_dim * 2]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
            cell: Final cell state [num_layers, batch_size, hidden_dim]
        """
        # [batch_size, src_len] -> [batch_size, src_len, embedding_dim]
        embedded = self.embedding(src)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)

        # outputs: [batch_size, src_len, hidden_dim * 2]
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # cell: [num_layers * 2, batch_size, hidden_dim]
        if lengths is not None:
            lengths = lengths.detach().to(device="cpu", dtype=torch.long)
            lengths = lengths.clamp(min=1, max=src.size(1))
            packed = pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, (hidden, cell) = self.rnn(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=src.size(1),
            )
        else:
            outputs, (hidden, cell) = self.rnn(embedded)
        outputs = self.output_norm(outputs)

        # Combine forward and backward hidden states
        # [num_layers * 2, batch_size, hidden_dim] -> [num_layers, batch_size, hidden_dim * 2]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)

        # [num_layers * 2, batch_size, hidden_dim] -> [num_layers, batch_size, hidden_dim * 2]
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)

        # Project hidden and cell to the right dimension
        # [num_layers, batch_size, hidden_dim * 2] -> [num_layers, batch_size, hidden_dim]
        hidden = self.fc(hidden)
        hidden = self.fc_norm(hidden)
        hidden = torch.tanh(hidden)

        # [num_layers, batch_size, hidden_dim * 2] -> [num_layers, batch_size, hidden_dim]
        cell = self.fc(cell)
        cell = self.fc_norm(cell)
        cell = torch.tanh(cell)

        return outputs, (hidden, cell)

class MultiHeadAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, n_heads, dropout=0.1):
        """
        Multi-Head Attention mechanism

        Args:
            enc_hidden_dim: Encoder's hidden dimension (per direction)
            dec_hidden_dim: Decoder's hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        assert dec_hidden_dim % n_heads == 0, "Hidden dimension must be divisible by number of heads"

        self.enc_hidden_dim = enc_hidden_dim * 2  # For bidirectional encoder
        self.dec_hidden_dim = dec_hidden_dim
        self.n_heads = n_heads 
        self.head_dim = dec_hidden_dim // n_heads

        # Linear layers for keys, queries, and values
        self.query_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        self.key_proj = nn.Linear(self.enc_hidden_dim, dec_hidden_dim)
        self.value_proj = nn.Linear(self.enc_hidden_dim, dec_hidden_dim)

        # Final output projection
        self.output_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)

        # Layer normalizations
        self.norm_q = LayerNorm(dec_hidden_dim)
        self.norm_k = LayerNorm(dec_hidden_dim)
        self.norm_v = LayerNorm(dec_hidden_dim)
        self.norm_out = LayerNorm(dec_hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Scale factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: Decoder hidden state [batch_size, dec_hidden_dim] or [batch_size, query_len, dec_hidden_dim]
            key_value: Encoder outputs [batch_size, src_len, enc_hidden_dim*2]
            mask: Optional mask for padding [batch_size, 1, src_len]

        Returns:
            context: Context vector [batch_size, dec_hidden_dim] or [batch_size, query_len, dec_hidden_dim]
            attention: Attention weights [batch_size, n_heads, query_len, src_len]
        """
        batch_size = key_value.shape[0]
        src_len = key_value.shape[1]

        # Reshape query if it's 2D
        is_2d = (len(query.shape) == 2)
        if is_2d:
            query = query.unsqueeze(1)  # [batch_size, 1, dec_hidden_dim]

        query_len = query.shape[1]

        # Linear projections and normalization
        # [batch_size, query_len, dec_hidden_dim]
        q = self.norm_q(self.query_proj(query))
        # [batch_size, src_len, dec_hidden_dim]
        k = self.norm_k(self.key_proj(key_value))
        # [batch_size, src_len, dec_hidden_dim]
        v = self.norm_v(self.value_proj(key_value))

        # Split into multiple heads
        # [batch_size, query_len, n_heads, head_dim]
        q = q.view(batch_size, query_len, self.n_heads, self.head_dim)
        # [batch_size, src_len, n_heads, head_dim]
        k = k.view(batch_size, src_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.n_heads, self.head_dim)

        # Transpose for attention computation
        # [batch_size, n_heads, query_len, head_dim]
        q = q.transpose(1, 2)
        # [batch_size, n_heads, src_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        # [batch_size, n_heads, query_len, src_len]
        energy = torch.matmul(q, k.transpose(2, 3)) / self.scale

        # Apply mask if provided
        if mask is not None:
            # Reshape mask for broadcasting: [batch_size, 1, 1, src_len]
            # This allows broadcasting across n_heads and query_len dimensions
            mask = mask.unsqueeze(1)
            mask_fill_value = -1e4 if energy.dtype == torch.float16 else -1e10
            energy = energy.masked_fill(mask == 0, mask_fill_value)

        # Apply softmax to get attention weights
        # [batch_size, n_heads, query_len, src_len]
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Apply attention weights to values
        # [batch_size, n_heads, query_len, head_dim]
        output = torch.matmul(attention, v)

        # Concatenate heads and reshape
        # [batch_size, query_len, n_heads, head_dim] -> [batch_size, query_len, dec_hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.dec_hidden_dim)

        # Final projection and normalization
        output = self.norm_out(self.output_proj(output))

        # Remove sequence dimension if input was 2D
        if is_2d:
            output = output.squeeze(1)
            attention = attention[:, :, 0, :]

        return output, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim,
                num_layers, dropout, n_heads):
        """
        LSTM Decoder with Multi-Head Attention

        Args:
            output_dim: Size of vocabulary
            embedding_dim: Dimension of embeddings
            enc_hidden_dim: Encoder's hidden dimension (per direction)
            dec_hidden_dim: Decoder's hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            n_heads: Number of attention heads
        """
        super().__init__()

        self.output_dim = output_dim
        self.dec_hidden_dim = dec_hidden_dim

        # Embedding layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.embed_norm = LayerNorm(embedding_dim)

        # Multi-head attention layer
        self.attention = MultiHeadAttention(enc_hidden_dim, dec_hidden_dim, n_heads, dropout)

        # LSTM layers
        self.rnn = nn.LSTM(embedding_dim + dec_hidden_dim,
                          dec_hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)

        self.rnn_norm = LayerNorm(dec_hidden_dim)

        # Output layer
        self.fc_out = nn.Linear(dec_hidden_dim + embedding_dim + dec_hidden_dim, output_dim)
        self.output_norm = LayerNorm(output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, src_mask=None):
        """
        Args:
            input: Current input token [batch_size, 1]
            hidden: Current hidden state [num_layers, batch_size, dec_hidden_dim]
            cell: Current cell state [num_layers, batch_size, dec_hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden_dim * 2]
            src_mask: Optional mask for padding [batch_size, 1, src_len]

        Returns:
            prediction: Output prediction [batch_size, output_dim]
            hidden: Updated hidden state [num_layers, batch_size, dec_hidden_dim]
            cell: Updated cell state [num_layers, batch_size, dec_hidden_dim]
            attention: Attention weights [batch_size, n_heads, src_len]
        """
        # Convert input to embedding
        # [batch_size, 1] -> [batch_size, 1, embedding_dim]
        embedded = self.embedding(input)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)

        # Use last layer hidden state for attention
        # [num_layers, batch_size, dec_hidden_dim] -> [batch_size, dec_hidden_dim]
        query = hidden[-1]

        # Get context vector and attention weights from multi-head attention
        # context: [batch_size, dec_hidden_dim]
        # attn_weights: [batch_size, n_heads, src_len]
        context, attn_weights = self.attention(query, encoder_outputs, src_mask)

        # Reshape context for concatenation
        # [batch_size, dec_hidden_dim] -> [batch_size, 1, dec_hidden_dim]
        context = context.unsqueeze(1)

        # Combine embedding and context vector for RNN input
        # [batch_size, 1, embedding_dim + dec_hidden_dim]
        rnn_input = torch.cat((embedded, context), dim=2)

        # Pass through RNN
        # output: [batch_size, 1, dec_hidden_dim]
        # hidden: [num_layers, batch_size, dec_hidden_dim]
        # cell: [num_layers, batch_size, dec_hidden_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = self.rnn_norm(output)

        # Prepare inputs for final prediction
        # [batch_size, 1, dec_hidden_dim + embedding_dim + dec_hidden_dim]
        prediction_input = torch.cat((output, embedded, context), dim=2)

        # Get prediction
        # [batch_size, 1, output_dim] -> [batch_size, output_dim]
        prediction = self.fc_out(prediction_input)
        prediction = self.output_norm(prediction)
        prediction = prediction.squeeze(1)

        return prediction, hidden, cell, attn_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Projection layers to match encoder's hidden_dim to decoder's hidden_dim
        self.hidden_proj = nn.Linear(encoder.hidden_dim, decoder.dec_hidden_dim)
        self.cell_proj = nn.Linear(encoder.hidden_dim, decoder.dec_hidden_dim)

        # Layer normalizations for projections
        self.hidden_norm = LayerNorm(decoder.dec_hidden_dim)
        self.cell_norm = LayerNorm(decoder.dec_hidden_dim)

    def make_src_mask(self, src, src_lengths=None):
        """
        Create a mask for src padding
        Args:
            src: Source tensor [batch_size, src_len]
            src_lengths: Optional unpadded source lengths [batch_size]
        Returns:
            mask: Mask tensor [batch_size, 1, src_len]
        """
        # src_mask: [batch_size, 1, src_len]
        if src_lengths is not None:
            src_lengths = src_lengths.to(device=src.device, dtype=torch.long)
            src_lengths = src_lengths.clamp(min=1, max=src.size(1))
            positions = torch.arange(src.size(1), device=src.device).unsqueeze(0)
            src_mask = (positions < src_lengths.unsqueeze(1)).unsqueeze(1)
        else:
            src_mask = (src != 0).unsqueeze(1)
        return src_mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        Args:
            src: Source tensor [batch_size, src_len]
            trg: Target tensor [batch_size, trg_len]
            teacher_forcing_ratio: Probability to use teacher forcing
            src_lengths: Optional unpadded source lengths [batch_size]

        Returns:
            outputs: Output predictions [batch_size, trg_len, output_dim]
            attention_weights: Attention weights [batch_size, trg_len, n_heads, src_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        src_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        n_heads = self.decoder.attention.n_heads

        # Initialize outputs tensor
        # [batch_size, trg_len, trg_vocab_size]
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        # [batch_size, trg_len, n_heads, src_len]
        attention_weights = torch.zeros(batch_size, trg_len, n_heads, src_len).to(self.device)

        # Create source mask for attention
        src_mask = self.make_src_mask(src, src_lengths)

        # Encode source sequence
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        # hidden: [num_layers, batch_size, enc_hidden_dim]
        # cell: [num_layers, batch_size, enc_hidden_dim]
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)

        # Project encoder's final hidden state to decoder's dimensions
        # [num_layers, batch_size, dec_hidden_dim]
        hidden = self.hidden_norm(self.hidden_proj(hidden))
        hidden = torch.tanh(hidden)

        # [num_layers, batch_size, dec_hidden_dim]
        cell = self.cell_norm(self.cell_proj(cell))
        cell = torch.tanh(cell)

        # First decoder input is the BOS token (first token of target sequence)
        # [batch_size, 1]
        input = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            # Get output from decoder
            # output: [batch_size, output_dim]
            # hidden: [num_layers, batch_size, dec_hidden_dim]
            # cell: [num_layers, batch_size, dec_hidden_dim]
            # attn_weights: [batch_size, n_heads, src_len]
            output, hidden, cell, attn_weights = self.decoder(
                input, hidden, cell, encoder_outputs, src_mask
            )

            # Store output and attention weights
            outputs[:, t, :] = output
            attention_weights[:, t, :, :] = attn_weights

            # Decide whether to teacher force or not
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token
            # [batch_size]
            top1 = output.argmax(1)

            # Use teacher forcing or model prediction
            # [batch_size, 1]
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs, attention_weights

def create_seq2seq_model(vocab_size, hidden_dim, embedding_dim, num_layers, dropout, n_heads, device):
    """
    Factory function to create a Seq2Seq model with multi-head attention.

    Args:
        vocab_size: Size of vocabulary (shared between source and target)
        hidden_dim: Size of the decoder's hidden dimension
        embedding_dim: Dimension of embeddings
        num_layers: Number of LSTM layers in encoder and decoder
        dropout: Dropout rate
        n_heads: Number of attention heads
        device: Device to run the model on

    Returns:
        model: Seq2SeqWithAttention model
    """
    # Calculate encoder hidden dimension (half of decoder since bidirectional)
    enc_hidden_dim = hidden_dim // 2
    dec_hidden_dim = hidden_dim

    # Create encoder and decoder
    encoder = Encoder(vocab_size, embedding_dim, enc_hidden_dim, num_layers, dropout)
    decoder = Decoder(vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim,
                    num_layers, dropout, n_heads)

    # Create Seq2Seq model
    model = Seq2SeqWithAttention(encoder, decoder, device)

    # Initialize parameters
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    return model.to(device)
=======
"""Seq2Seq model with bidirectional LSTM encoder-decoder, layer normalization and multi-head attention."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import LayerNorm


# class LayerNorm(nn.Module):
#     """Layer Normalization module"""
#     def __init__(self, features, eps=1e-6):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(features))
#         self.beta = nn.Parameter(torch.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         # x: [batch_size, seq_len, features] or [batch_size, features]
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Bidirectional LSTM Encoder with layer normalization
        
        Args:
            input_dim: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Size of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embed_norm = LayerNorm(embedding_dim)
        
        self.rnn = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=num_layers, 
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        
        self.output_norm = LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: Source tensor [batch_size, src_len]
            
        Returns:
            outputs: Output features [batch_size, src_len, hidden_dim * 2]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
            cell: Final cell state [num_layers, batch_size, hidden_dim]
        """
        # [batch_size, src_len] -> [batch_size, src_len, embedding_dim]
        embedded = self.embedding(src)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)
        
        if src_lengths is not None:
            # Pack to skip padded positions
            packed = pack_padded_sequence(
                embedded,
                src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, (hidden, cell) = self.rnn(packed)
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            # Fallback: no packing
            outputs, (hidden, cell) = self.rnn(embedded)
        
        outputs = self.output_norm(outputs)
        
        # Combine forward and backward hidden states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        
        # Project hidden and cell to the right dimension
        hidden = self.fc(hidden)
        hidden = self.fc_norm(hidden)
        hidden = torch.tanh(hidden)
        
        cell = self.fc(cell)
        cell = self.fc_norm(cell)
        cell = torch.tanh(cell)
        
        return outputs, (hidden, cell)


class MultiHeadAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, n_heads, dropout=0.1):
        """
        Multi-Head Attention mechanism
        
        Args:
            enc_hidden_dim: Encoder's hidden dimension (per direction)
            dec_hidden_dim: Decoder's hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert dec_hidden_dim % n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.enc_hidden_dim = enc_hidden_dim * 2  # For bidirectional encoder
        self.dec_hidden_dim = dec_hidden_dim
        self.n_heads = n_heads
        self.head_dim = dec_hidden_dim // n_heads
        
        # Linear layers for keys, queries, and values
        self.query_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        self.key_proj = nn.Linear(self.enc_hidden_dim, dec_hidden_dim)
        self.value_proj = nn.Linear(self.enc_hidden_dim, dec_hidden_dim)
        
        # Final output projection
        self.output_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        
        # Layer normalizations
        self.norm_q = LayerNorm(dec_hidden_dim)
        self.norm_k = LayerNorm(dec_hidden_dim)
        self.norm_v = LayerNorm(dec_hidden_dim)
        self.norm_out = LayerNorm(dec_hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()
    
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: Decoder hidden state [batch_size, dec_hidden_dim] or [batch_size, query_len, dec_hidden_dim]
            key_value: Encoder outputs [batch_size, src_len, enc_hidden_dim*2]
            mask: Optional mask for padding [batch_size, 1, src_len]
            
        Returns:
            context: Context vector [batch_size, dec_hidden_dim] or [batch_size, query_len, dec_hidden_dim]
            attention: Attention weights [batch_size, n_heads, query_len, src_len]
        """
        batch_size = key_value.shape[0]
        src_len = key_value.shape[1]
        
        # Reshape query if it's 2D
        is_2d = (len(query.shape) == 2)
        if is_2d:
            query = query.unsqueeze(1)  # [batch_size, 1, dec_hidden_dim]
        
        query_len = query.shape[1]
        
        # Linear projections and normalization
        # [batch_size, query_len, dec_hidden_dim]
        q = self.norm_q(self.query_proj(query))
        # [batch_size, src_len, dec_hidden_dim]
        k = self.norm_k(self.key_proj(key_value))
        # [batch_size, src_len, dec_hidden_dim]
        v = self.norm_v(self.value_proj(key_value))
        
        # Split into multiple heads
        # [batch_size, query_len, n_heads, head_dim]
        q = q.view(batch_size, query_len, self.n_heads, self.head_dim)
        # [batch_size, src_len, n_heads, head_dim]
        k = k.view(batch_size, src_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        # [batch_size, n_heads, query_len, head_dim]
        q = q.transpose(1, 2)
        # [batch_size, n_heads, src_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        # [batch_size, n_heads, query_len, src_len]
        energy = torch.matmul(q, k.transpose(2, 3)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask for broadcasting: [batch_size, 1, 1, src_len]
            # This allows broadcasting across n_heads and query_len dimensions
            mask = mask.unsqueeze(1)
            mask_fill_value = -1e4 if energy.dtype == torch.float16 else -1e10
            energy = energy.masked_fill(mask == 0, mask_fill_value)
        
        # Apply softmax to get attention weights
        # [batch_size, n_heads, query_len, src_len]
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        # [batch_size, n_heads, query_len, head_dim]
        output = torch.matmul(attention, v)
        
        # Concatenate heads and reshape
        # [batch_size, query_len, n_heads, head_dim] -> [batch_size, query_len, dec_hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.dec_hidden_dim)
        
        # Final projection and normalization
        output = self.norm_out(self.output_proj(output))
        
        # Remove sequence dimension if input was 2D
        if is_2d:
            output = output.squeeze(1)
            attention = attention[:, :, 0, :]
        
        return output, attention
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, 
                num_layers, dropout, n_heads):
        """
        LSTM Decoder with Multi-Head Attention
        
        Args:
            output_dim: Size of vocabulary
            embedding_dim: Dimension of embeddings
            enc_hidden_dim: Encoder's hidden dimension (per direction)
            dec_hidden_dim: Decoder's hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            n_heads: Number of attention heads
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.dec_hidden_dim = dec_hidden_dim
        
        # Embedding layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.embed_norm = LayerNorm(embedding_dim)
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(enc_hidden_dim, dec_hidden_dim, n_heads, dropout)
        
        # LSTM layers
        self.rnn = nn.LSTM(embedding_dim + dec_hidden_dim, 
                          dec_hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        
        self.rnn_norm = LayerNorm(dec_hidden_dim)
        
        # Output layer
        self.fc_out = nn.Linear(dec_hidden_dim + embedding_dim + dec_hidden_dim, output_dim)
        self.output_norm = LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, src_mask=None):
        """
        Args:
            input: Current input token [batch_size, 1]
            hidden: Current hidden state [num_layers, batch_size, dec_hidden_dim]
            cell: Current cell state [num_layers, batch_size, dec_hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden_dim * 2]
            src_mask: Optional mask for padding [batch_size, 1, src_len]
            
        Returns:
            prediction: Output prediction [batch_size, output_dim]
            hidden: Updated hidden state [num_layers, batch_size, dec_hidden_dim]
            cell: Updated cell state [num_layers, batch_size, dec_hidden_dim]
            attention: Attention weights [batch_size, n_heads, src_len]
        """
        # Convert input to embedding
        # [batch_size, 1] -> [batch_size, 1, embedding_dim]
        embedded = self.embedding(input)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)
        
        # Use last layer hidden state for attention
        # [num_layers, batch_size, dec_hidden_dim] -> [batch_size, dec_hidden_dim]
        query = hidden[-1]
        
        # Get context vector and attention weights from multi-head attention
        # context: [batch_size, dec_hidden_dim]
        # attn_weights: [batch_size, n_heads, src_len]
        context, attn_weights = self.attention(query, encoder_outputs, src_mask)
        
        # Reshape context for concatenation
        # [batch_size, dec_hidden_dim] -> [batch_size, 1, dec_hidden_dim]
        context = context.unsqueeze(1)
        
        # Combine embedding and context vector for RNN input
        # [batch_size, 1, embedding_dim + dec_hidden_dim]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Pass through RNN
        # output: [batch_size, 1, dec_hidden_dim]
        # hidden: [num_layers, batch_size, dec_hidden_dim]
        # cell: [num_layers, batch_size, dec_hidden_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = self.rnn_norm(output)
        
        # Prepare inputs for final prediction
        # [batch_size, 1, dec_hidden_dim + embedding_dim + dec_hidden_dim]
        prediction_input = torch.cat((output, embedded, context), dim=2)
        
        # Get prediction
        # [batch_size, 1, output_dim] -> [batch_size, output_dim]
        prediction = self.fc_out(prediction_input)
        prediction = self.output_norm(prediction)
        prediction = prediction.squeeze(1)
        
        return prediction, hidden, cell, attn_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Projection layers to match encoder's hidden_dim to decoder's hidden_dim
        self.hidden_proj = nn.Linear(encoder.hidden_dim, decoder.dec_hidden_dim)
        self.cell_proj = nn.Linear(encoder.hidden_dim, decoder.dec_hidden_dim)
        
        # Layer normalizations for projections
        self.hidden_norm = LayerNorm(decoder.dec_hidden_dim)
        self.cell_norm = LayerNorm(decoder.dec_hidden_dim)
        
    def make_src_mask(self, src):
        """
        Create a mask for src padding
        Args:
            src: Source tensor [batch_size, src_len]
        Returns:
            mask: Mask tensor [batch_size, 1, src_len]
        """
        # src_mask: [batch_size, 1, src_len]
        src_mask = (src != 0).unsqueeze(1)
        return src_mask
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        Args:
            src: Source tensor [batch_size, src_len]
            trg: Target tensor [batch_size, trg_len]
            teacher_forcing_ratio: Probability to use teacher forcing
            
        Returns:
            outputs: Output predictions [batch_size, trg_len, output_dim]
            attention_weights: Attention weights [batch_size, trg_len, n_heads, src_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        src_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        n_heads = self.decoder.attention.n_heads
        
        # Initialize outputs tensor
        # [batch_size, trg_len, trg_vocab_size]
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        # [batch_size, trg_len, n_heads, src_len]
        attention_weights = torch.zeros(batch_size, trg_len, n_heads, src_len).to(self.device)
        
        # Create source mask for attention
        src_mask = self.make_src_mask(src)
        
        # Encode source sequence
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        # hidden: [num_layers, batch_size, enc_hidden_dim]
        # cell: [num_layers, batch_size, enc_hidden_dim]
        # encoder_outputs, (hidden, cell) = self.encoder(src) 
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths=src_lengths)

        
        # Project encoder's final hidden state to decoder's dimensions
        # [num_layers, batch_size, dec_hidden_dim]
        hidden = self.hidden_norm(self.hidden_proj(hidden))
        hidden = torch.tanh(hidden)
        
        # [num_layers, batch_size, dec_hidden_dim]
        cell = self.cell_norm(self.cell_proj(cell))
        cell = torch.tanh(cell)
        
        # First decoder input is the BOS token (first token of target sequence)
        # [batch_size, 1]
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            # Get output from decoder
            # output: [batch_size, output_dim]
            # hidden: [num_layers, batch_size, dec_hidden_dim]
            # cell: [num_layers, batch_size, dec_hidden_dim]
            # attn_weights: [batch_size, n_heads, src_len]
            output, hidden, cell, attn_weights = self.decoder(
                input, hidden, cell, encoder_outputs, src_mask
            )
            
            # Store output and attention weights
            outputs[:, t, :] = output
            attention_weights[:, t, :, :] = attn_weights
            
            # Decide whether to teacher force or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            # [batch_size]
            top1 = output.argmax(1)
            
            # Use teacher forcing or model prediction
            # [batch_size, 1]
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attention_weights

def create_seq2seq_model(vocab_size, hidden_dim, embedding_dim, num_layers, dropout, n_heads, device):
    """
    Factory function to create a Seq2Seq model with multi-head attention.
    
    Args:
        vocab_size: Size of vocabulary (shared between source and target)
        hidden_dim: Size of the decoder's hidden dimension
        embedding_dim: Dimension of embeddings
        num_layers: Number of LSTM layers in encoder and decoder
        dropout: Dropout rate
        n_heads: Number of attention heads
        device: Device to run the model on
        
    Returns:
        model: Seq2SeqWithAttention model
    """
    # Calculate encoder hidden dimension (half of decoder since bidirectional)
    enc_hidden_dim = hidden_dim // 2
    dec_hidden_dim = hidden_dim
    
    # Create encoder and decoder
    encoder = Encoder(vocab_size, embedding_dim, enc_hidden_dim, num_layers, dropout)
    decoder = Decoder(vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, 
                    num_layers, dropout, n_heads)
    
    # Create Seq2Seq model
    model = Seq2SeqWithAttention(encoder, decoder, device)
    
    # Initialize parameters
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model.to(device)
>>>>>>> origin/master
