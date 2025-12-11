"""Seq2Seq model with bidirectional LSTM encoder-decoder and attention."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Bidirectional LSTM Encoder
        
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
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Args:
            src: Source tensor [batch_size, src_len]
            
        Returns:
            outputs: Output features [batch_size, src_len, hidden_dim * 2]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
            cell: Final cell state [num_layers, batch_size, hidden_dim]
        """
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embedding_dim]
        
        # outputs: [batch_size, src_len, hidden_dim * 2]
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # cell: [num_layers * 2, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Combine forward and backward hidden states
        # hidden: [num_layers * 2, batch_size, hidden_dim] -> [num_layers, batch_size, hidden_dim * 2]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        
        # cell: [num_layers * 2, batch_size, hidden_dim] -> [num_layers, batch_size, hidden_dim * 2]
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        
        # Project hidden and cell to the right dimension
        hidden = torch.tanh(self.fc(hidden))  # [num_layers, batch_size, hidden_dim]
        cell = torch.tanh(self.fc(cell))      # [num_layers, batch_size, hidden_dim]
        
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        """
        Attention mechanism
        
        Args:
            enc_hidden_dim: Encoder's hidden dimension
            dec_hidden_dim: Decoder's hidden dimension
        """
        super().__init__()
        
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: Current decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden_dim * 2]
            
        Returns:
            attention: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # hidden: [batch_size, src_len, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        # energy: [batch_size, src_len, dec_hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Get attention scores
        # attention: [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, 
                 num_layers, dropout, attention):
        """
        LSTM Decoder with Attention
        
        Args:
            output_dim: Size of vocabulary
            embedding_dim: Dimension of embeddings
            enc_hidden_dim: Encoder's hidden dimension
            dec_hidden_dim: Decoder's hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention: Attention module
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        
        self.rnn = nn.LSTM((enc_hidden_dim * 2) + embedding_dim, 
                          dec_hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        """
        Args:
            input: Current input token [batch_size, 1]
            hidden: Current hidden state [num_layers, batch_size, hidden_dim]
            cell: Current cell state [num_layers, batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden_dim * 2]
            
        Returns:
            prediction: Output prediction [batch_size, output_dim]
            hidden: Updated hidden state
            cell: Updated cell state
            attention: Attention weights
        """
        # Convert input to embedding
        # input: [batch_size, 1] -> embedded: [batch_size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input))
        
        # Get attention weights from last layer hidden state
        # hidden[-1]: [batch_size, hidden_dim]
        # a: [batch_size, src_len]
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Apply attention weights to encoder outputs
        # weighted: [batch_size, 1, enc_hidden_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        
        # Concatenate embedding and weighted context vector
        # rnn_input: [batch_size, 1, (enc_hidden_dim * 2) + embedding_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        # Pass through RNN
        # output: [batch_size, seq_len, dec_hidden_dim]
        # hidden: [num_layers, batch_size, dec_hidden_dim]
        # cell: [num_layers, batch_size, dec_hidden_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Get prediction
        # embedded: [batch_size, 1, embedding_dim]
        # output: [batch_size, 1, dec_hidden_dim]
        # weighted: [batch_size, 1, enc_hidden_dim * 2]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        prediction = prediction.squeeze(1)  # [batch_size, output_dim]
        
        return prediction, hidden, cell, a.squeeze(1)

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # Projection layers to match encoder's hidden_dim to decoder's hidden_dim
        self.hidden_proj = nn.Linear(encoder.hidden_dim, decoder.rnn.hidden_size)
        self.cell_proj = nn.Linear(encoder.hidden_dim, decoder.rnn.hidden_size)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Project encoder's hidden and cell states to decoder's hidden dimension
        hidden = torch.tanh(self.hidden_proj(hidden))  # [num_layers, batch_size, dec_hidden_dim]
        cell = torch.tanh(self.cell_proj(cell))        # [num_layers, batch_size, dec_hidden_dim]
        
        # First input is <BOS>
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            attentions[:, t, :] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions

def create_seq2seq_model(vocab_size, hidden_dim, embedding_dim, num_layers, dropout, device):
    """Factory function to create a Seq2Seq model with attention."""
    enc_hidden_dim = hidden_dim // 2  # Since we're using bidirectional encoder
    dec_hidden_dim = hidden_dim
    
    # Create attention module
    attention = Attention(enc_hidden_dim, dec_hidden_dim)
    
    # Create encoder and decoder
    encoder = Encoder(vocab_size, embedding_dim, enc_hidden_dim, num_layers, dropout)
    decoder = Decoder(vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, 
                      num_layers, dropout, attention)
    
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