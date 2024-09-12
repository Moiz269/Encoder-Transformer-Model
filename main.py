import torch
import torch.nn as nn
import torch.optim as optim
from utils import replicate
from attention import MultiHeadAttention
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
from encoder import TransformerBlock
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from datasets import load_dataset
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Sequence, Strip
import pandas as pd
import sys
import csv
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class CustomTextDataset(Dataset):
    def __init__(self, csv_file):
        # self.text_data = text_data
        #self.vocab = vocab
        # self.tokenizer = tokenizer
        self.code_snippets_df = pd.read_csv(csv_file)

        # Convert the merged column into a list of strings
        self.code_snippets = self.code_snippets_df['merged'].tolist()
            
        self.tokenizer = self.prepare_tokenizer()
                   
    def prepare_tokenizer(self): 
        # Increase field size limit to handle large CSV fields
        csv.field_size_limit(12456763)

        # Initialize a tokenizer for Python code snippets
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        # Normalizer: Strip extra spaces (no lowercase since Python is case-sensitive)
        tokenizer.normalizer = Sequence([Strip()])

        # Pre-Tokenizer: Split based on whitespace and handle byte-level characters
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])

        # Tokenizer Trainer: Train on the Python code snippets
        trainer = trainers.WordPieceTrainer(
            vocab_size=50000,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )

        # Train the tokenizer on the merged code snippets
        tokenizer.train_from_iterator(self.code_snippets, trainer=trainer)

        # Post-processing: Adding [CLS] and [SEP] tokens for sequence processing
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]"))
            ]
        )
        tokenizer.enable_padding(length=MAX_SEQ_LENGTH)
        tokenizer.enable_truncation(MAX_SEQ_LENGTH)
        return tokenizer
    

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, idx):
        text = self.code_snippets[idx]
        tokens = self.tokenizer.encode(text).tokens
        token_ids = torch.tensor([self.tokenizer.get_vocab().get(token, "[UNK]") for token in tokens], dtype=torch.long)
        # Return the input (src) and a shifted version as target (tgt)
        return token_ids[:-1], token_ids[1:]  # src, tgt


csv_file = (r'C:/Users/aiint/Downloads/modified_python_code_dataset.csv')

# Define the path where the model is saved
model_file = 'model.json'

# Define TensorBoard log directory
log_dir = "runs/transformer_experiment"

# Create a TensorBoard SummaryWriter
writer = SummaryWriter(log_dir)

# Check if the model file exists
if os.path.exists(model_file):
    print("Loading the pre-trained model...")
    model = torch.load(model_file)

else:
    print("Training the model from scratch...")

    EMBEDDING_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 2048
    NUM_DECODER_LAYERS = 6
    MAX_SEQ_LENGTH = 100
    VOCAB_SIZE = 50000

    class DecoderBlock(nn.Module):
        def __init__(self, embed_dim=512, heads=8, expansion_factor=4, dropout=0.2):
            super(DecoderBlock, self).__init__()
            
            # First define the Decoder Multi-head attention
            self.attention = MultiHeadAttention(embed_dim, heads)
            # normalization
            self.norm = nn.LayerNorm(embed_dim)
            # Dropout to avoid overfitting
            self.dropout = nn.Dropout(dropout)
            # Finally the transformerBlock
            self.transformerBlock = TransformerBlock(embed_dim, heads, expansion_factor, dropout)

        def forward(self, key, query, x, mask):
            # Pass the inputs to the decoder multi-head attention
            decoder_attention = self.attention(x, x, x, mask)
            # Residual connection + normalization
            value = self.dropout(self.norm(decoder_attention + x))
            # Return the value (output after attention and normalization)
            return value

    class Decoder(nn.Module):
        def __init__(self, target_vocab_size, seq_len, embed_dim=512, num_blocks=6, expansion_factor=4, heads=8, dropout=0.2):
            """
            The Decoder part of the Transformer architecture.

            It is a set of stacked decoders on top of each other. In the paper, they used a stack of 6 decoders.
            """
            super(Decoder, self).__init__()

            # Define the embedding
            self.embedding = nn.Embedding(target_vocab_size, embed_dim)
            # The positional embedding
            self.positional_encoder = PositionalEncoding(embed_dim, seq_len)
            # Define the set of decoders
            self.blocks = nn.ModuleList([DecoderBlock(embed_dim, heads, expansion_factor, dropout) for _ in range(num_blocks)])
            # Dropout for overfitting
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
            x = self.dropout(x)  # 32x10x512

            for block in self.blocks:
                x = block(x, x, x, mask)

            return x

    class PositionalEncoding(nn.Module):
        def __init__(self, embedding_dim, max_seq_length):
            super(PositionalEncoding, self).__init__()
            pe = torch.zeros(max_seq_length, embedding_dim)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return x
    
    class TransformerModel(nn.Module):
        def __init__(self, VOCAB_SIZE):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
            self.pos_encoder = PositionalEncoding(EMBEDDING_SIZE, MAX_SEQ_LENGTH)
            self.transformer_decoder = Decoder(VOCAB_SIZE, MAX_SEQ_LENGTH, EMBEDDING_SIZE, heads=NHEAD)
            self.fc_out = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)
            self.softmax = nn.Softmax(dim=2)

        def generate_square_subsequent_mask(self, sz):
            mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
            return mask

        def forward(self, src):
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
            src = self.embedding(src) * math.sqrt(EMBEDDING_SIZE)
            src = self.pos_encoder(src)
            output = self.transformer_decoder(src, src_mask)
            output = self.fc_out(output)
            output = self.softmax(output)
            return output

    model = TransformerModel(VOCAB_SIZE)
    
    # DATA SPLITTING
    
    dataset = CustomTextDataset(csv_file)
    train_ratio=0.7
    val_ratio=0.1
    test_ratio=0.2
    
    # Split dataset into train and temp (temp= train + test)
    
    train_size=int(train_ratio*len(dataset))
    temp_size=len(dataset)-train_size
    train_dataset, temp_dataset=random_split(dataset,[train_size, temp_size])
    
    # Split temp data into validation and test sets
    val_size=int(val_ratio/(val_ratio+test_ratio)*temp_size)
    test_size=temp_size-val_size
    val_dataset, test_dataset=random_split(temp_dataset,[val_size,test_size])
    
    BATCH_SIZE = 16
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # TRAINING LOOP

    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    dataset = CustomTextDataset(csv_file)  # Assuming dataset is well defined
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=True)

        for i, (src, tgt) in progress_bar:
            optimizer.zero_grad()

            # Forward pass
            output = model(src)

            # Compute the loss
            loss = criterion(output.view(-1, VOCAB_SIZE), tgt.reshape(-1))

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate average loss and perplexity
        avg_train_loss = total_loss / len(train_loader)
        train_perplexity = math.exp(avg_train_loss)

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
        writer.add_scalar("Perplexity/Train", train_perplexity, epoch+1)
        
        # VALIDATION LOOP
        
        model.eval()
        total_val_loss=0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                output=model(src)
                val_loss=criterion(output.view(-1, VOCAB_SIZE), tgt.reshape(-1))
                total_val_loss += val_loss.item()
                
        # Calculate avg loss and perplexity to TB
        avg_val_loss = total_val_loss / len(val_loader)
        val_perplexity = math.exp(avg_val_loss)
        
        # Log validation loss and perplexity to TensorBoard
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch+1)
        writer.add_scalar("Perplexity/Validation", val_perplexity, epoch+1)

        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}')
            
    # Save the trained model to a file
    torch.save(model, model_file)

# Close TensorBoard writer
writer.close()

    # Load the model from the saved file
    # model = model.from_file("model.json")
