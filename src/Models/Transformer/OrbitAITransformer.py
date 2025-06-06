import torch
import torch.nn as nn
from scipy.signal import max_len_seq

#Hyperparameters:
'''
Explanations (as needed):

    EMBED_DIM: the embedding dimension is the size of the transformed input features before they
    pass through the attention layers.
    
    Having a higher dimension allows the transformer to capture more nuanced interactions and relationships
    between values in the input sequence. 128 dimensions is a common choice, balancing learning accuracy and compute
    
    FEED_FORWARD_DIM: the number of layers in the MLP (feed-forward network) is commonly between 2-4 larger than the
    embedding layer. Increasing the dimensions increases complexity and learning power.
    
     
'''

'''
Embedding Layer: Since our input consists of 6 continuous features:
(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z) we will project this into a higher-dimensional space using a fully connected
layer.
This will help the model learn feature representations that capture relationships between input variables.

'''
class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(InputEmbedding, self).__init__() #initializes the parent module (nn.Module)
        self.linear = nn.Linear(input_dim, embed_dim) #Linear layer to project input into embedding space.

    def forward(self, x):
        return self.linear(x)

'''
Positional Encoding Layer (Based on time stamps)
'''
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(LearnedPositionalEncoding, self).__init__()
        #One positional vector per sequence position
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

    def forward(self, x):
        '''
        x:Tensor of shape (BATCH_SIZE, SEQ_LENGTH, EMBED_DIM)
        '''
        seq_length = x.size(1)

        #By slicing with the following code, we can select embeddings that exactly match the current input sequence length
        x = x + self.pos_embedding[:, :seq_length, :]
        return x
'''
Transformer Encoder

For both the Encoder and Decoder, the built-in PyTorch model includes:
Multi-head self-attention
Layer Norm
Skip Connections
Feed Forward MLPs
'''
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers,dropout):
        super(TransformerEncoder, self).__init__()

        #Single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = feedforward_dim,
            dropout = dropout,
            batch_first = True #PyTorch expects [sequence_length, batch, embed] shaped tensor by default
        )

        #Now to create a stack of encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        '''
        src: input data, tensor of shape (batch_size, sequence_length, embedding_dimensions)
        returns: (batch_size, sequence_length, embedding_dimension)
        '''

        #Pass through transformer encoder for prediction
        encoded_data = self.encoder(src)

        return encoded_data
'''
Transformer Decoder
'''
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, dropout):
        super(TransformerDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = feedforward_dim,
            dropout = dropout,
            batch_first = True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)

    def forward(self, tgt, memory):
        '''

        tgt: a typical name for the input sequence being sent into a decoder, short for target
        memory: (batch, src_sequence_length, embed_dim) - this is output from the encoder
        '''
        #Decode!
        out = self.decoder(tgt=tgt, memory=memory)

        return out
'''
Output Layer
input shape: [batch_size, sequence_length, embedding_dimensions]
output shape: [batch_size, sequence_length, 6] for the future state vectors
'''
class OutputProjection(nn.Module):
    def __init__(self, embed_dim):
        super(OutputProjection,self).__init__()
        #Fully connected MLP layer
        self.position_head = nn.Linear(embed_dim, 3) #Predict positions
        self.velocity_head = nn.Linear(embed_dim, 3) #Predict velocities

    def forward(self,x):
        position = self.position_head(x)
        velocity = self.velocity_head(x)
        return torch.cat([position,velocity], dim=-1) #[batch, seq_len,6]

'''
OrbitAI Transformer Model
'''
class OrbitAI(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_heads, feedforward_dim, num_layers, dropout, seq_len, pred_len):
        super(OrbitAI, self).__init__()

        self.embedding = InputEmbedding(
            input_dim = input_dim,
            embed_dim = embed_dim
            )

        self.src_encoded = LearnedPositionalEncoding(seq_len, embed_dim)
        self.tgt_encoded = LearnedPositionalEncoding(seq_len, embed_dim)

        self.encoder = TransformerEncoder(
            embed_dim = embed_dim,
            num_heads = num_heads,
            feedforward_dim = feedforward_dim,
            num_layers = num_layers,
            dropout = dropout
        )

        self.decoder = TransformerDecoder(
            embed_dim = embed_dim,
            num_heads = num_heads,
            feedforward_dim = feedforward_dim,
            num_layers = num_layers,
            dropout = dropout
        )

        self.output_layer = OutputProjection(embed_dim)

    def forward(self, src, tgt):
        '''
        src: [batch_size, src_seq_len, input_dim] (from the encoder)
        tgt: [batch_size, tgt_seq_len, input_dim] (from the decoder)
        '''
        src_embedded = self.embedding(src)
        src_encoded = self.src_encoded(src_embedded)

        tgt_embedded = self.embedding(tgt)
        tgt_encoded = self.tgt_encoded(tgt_embedded)

        #Transformer encoder
        memory = self.encoder(src_encoded)

        #Transformer decoder
        decoded = self.decoder(tgt_encoded, memory)

        #Capture attention weights for visualization during the forward pass
        out=decoded

        return self.output_layer(out)
