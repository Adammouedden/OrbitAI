import torch.optim as optim
import torch.nn as nn
import random
from OrbitAITransformer import OrbitAI
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

torch.cuda.empty_cache()


INPUT_DIM = 7          #The dimensions/neurons for the input layer: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
EMBED_DIM = 128        #Embedding Dimension for input vectors.
NUM_HEADS = 8          #Number of attention heads in multi-head attention block
NUM_LAYERS = 4         #Number of encoder layers
FEED_FORWARD_DIM = 256 #Size of feedforward layers within the Transformer's MLP
OUTPUT_DIM = 6         #Predicting the 6 dimensional outputs (the next state vectors)
LEARNING_RATE = 0.00001  #The learning rate for the optimizer function
BATCH_SIZE = 16       #Number of sequences per batch
EPOCHS = 50            #Number of training iterations
DROPOUT = 0.2          #Overfitting prevention
SEQ_LENGTH = 270        #Length of the input sequences, 270 = 8100/30 = PropagationDuration/steps
PRED_LEN = 180          #Number of sequences we want outputted
WEIGHT_DECAY = 1e-5     #Add weight decay to decrease importance of oversized values

def train(model, dataloader, optimizer, criterion, device, scheduled_sampling_ratio=0.0):
    model.train()
    total_position_loss = 0
    total_velocity_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_y = batch['tgt_y'].to(device)


        optimizer.zero_grad()

        #Start the decoder with the last frame of src
        decoder_input = tgt[:,:1,:] #shape:[batch,1,7]
        outputs = []

        for t in range(tgt_y.size(1)): #loop over each prediction step
            out = model(src,decoder_input) #output: [batch,cur_len,6]
            pred_delta = out[:, -1:, :] #last prediction [batch,1,6]
            true_delta = tgt_y[:, t:t+1, :] #ground truth delta at this step

            #Get previous full state (pos + vel only, 6D)
            last_state = decoder_input[:,-1:, 1:]

            #Dcide wheter to use model prediction or teacher forcing
            use_pred = random.random() < scheduled_sampling_ratio
            delta_to_use = pred_delta if use_pred else true_delta
            next_state = last_state + delta_to_use #apply delta to previous state

            #Advance time
            next_time = decoder_input[:,-1:,0:1] + delta_t
            #Build next decoder input
            next_input = torch.cat([next_time,next_state], dim=-1)#[batch,1,7]
            decoder_input = torch.cat([decoder_input,next_input],dim=1)

            outputs.append(pred_delta) #collect predictions

        output = torch.cat(outputs, dim=1) #batch, pred_len,6]

        #Separate predictions
        predict_position = output[:,:,:3]
        predict_velocity = output[:,:,3:]

        #Separate ground truths
        true_position = tgt_y[:,:,:3]
        true_velocity = tgt_y[:,:,3:]

        position_loss = criterion(predict_position,true_position)
        velocity_loss = criterion(predict_velocity, true_velocity)

        #Combined loss
        loss = position_loss + velocity_loss

        loss.backward()
        optimizer.step()

        total_position_loss += position_loss.item()
        total_velocity_loss += velocity_loss.item()


    n = len(dataloader)
    return total_position_loss / n, total_velocity_loss /n

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_position_loss = 0
    total_velocity_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_y = batch['tgt_y'].to(device)

            output = model(src, tgt)

            # Separate predictions
            predict_position = output[:, :, :3]
            predict_velocity = output[:, :, 3:]

            # Separate ground truths
            true_position = tgt_y[:, :, :3]
            true_velocity = tgt_y[:, :, 3:]

            position_loss = criterion(predict_position, true_position)
            velocity_loss = criterion(predict_velocity, true_velocity)

            total_position_loss += position_loss.item()
            total_velocity_loss += velocity_loss.item()

    n = len(dataloader)
    return total_position_loss/n, total_velocity_loss/n


class OrbitDataset(Dataset):
    def __init__(self, csv_path, input_len = 270, pred_len = 180, val_ratio = 0.2, split='train'):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        #Load the pre-processed CSV (time must be in seconds (float))
        df = pd.read_csv(csv_path)

        #SWtore full unnormalized data for later
        self.raw_df = df.copy()

        #Keep columns in order
        raw_data = df[['time', 'position_x', 'position_y', 'position_z',
                                'velocity_x', 'velocity_y', 'velocity_z']].dropna().values



        # Normalize these values with separate scalers
        self.time_scaler = StandardScaler()
        self.position_scaler = StandardScaler()
        self.velocity_scaler = StandardScaler()

        # Splitting the vectors and time
        self.time = raw_data[:, 0].reshape(-1, 1)
        self.position = raw_data[:, 1:4]  # [x,y,z]
        self.velocity = raw_data[:, 4:7]  # [vel_x, vel_y, vel_z]

        #Transform separated data
        time_scaled = self.time_scaler.fit_transform(self.time)
        position_scaled = self.position_scaler.fit_transform(self.position)
        velocity_scaled = self.velocity_scaler.fit_transform(self.velocity)

        #Reconstruct normalized full output [time, pos, vel]
        self.data = np.hstack([time_scaled, position_scaled, velocity_scaled])

        #4/10/2025 Change #002
        #Compute the state vector deltas and normalize them
        position_deltas = self.position[1:] - self.position[:-1]
        velocity_deltas = self.velocity[1:] - self.velocity[:-1]

        #print("Raw position units (first):", self.position[:3])

        self.delta_position_scaler = StandardScaler()
        self.delta_velocity_scaler = StandardScaler()

        self.delta_position_scaler.fit(position_deltas)
        self.delta_velocity_scaler.fit(velocity_deltas)


        #Split the data into train/test. No shuffling because this is temporal data
        train_data, val_data = train_test_split(self.data, test_size=val_ratio, shuffle=False)

        #4/10/2025 CHANGE 001
        self.normalized_time_step = self.time_scaler.transform([[30.0]])[0][0] - self.time_scaler.transform([[0.0]])[0][0]

        #Set data based on split
        if split == 'train':
            self.data = train_data
        else:
            self.data = val_data

    def __len__(self):
        return len(self.data) - (self.input_len + self.pred_len) + 1

    def __getitem__(self, idx):
        #Encoder input
        src = self.data[idx:idx + self.input_len]

        #Decoder input (use the last state from src then roll forward)
        tgt = self.data[idx + self.input_len - 1:idx + self.input_len + self.pred_len - 1]

        #Get future and previous state vectors
        future = self.data[idx + self.input_len:idx+self.input_len+self.pred_len,1:]
        previous = self.data[idx+self.input_len - 1: idx + self.input_len + self.pred_len -1, 1:]

        #4/10/2025 Change #002
        #Normalize the delta targets to be fed into the decoder
        raw_delta = future - previous
        delta_position = self.delta_position_scaler.transform(raw_delta[:, :3])
        delta_velocity = self.delta_velocity_scaler.transform(raw_delta[:, 3:])
        tgt_y = np.hstack([delta_position, delta_velocity])

        #Convert to tensors and then return
        '''
        src = [10,7]
        tgt = [5,7]
        tgt_y = [5,6]
        '''
        return{
            'src': torch.tensor(src, dtype=torch.float32),
            'tgt': torch.tensor(tgt, dtype=torch.float32),
            'tgt_y': torch.tensor(tgt_y, dtype=torch.float32)
        }

    def inverse_transform_deltas(self, delta_prediction):

        #Reverse scaling of predicted state vectors, un-normalized them
        delta_position = delta_prediction[:, :3]
        delta_velocity = delta_prediction[:, 3:]

        #4/10/2025 Change #002 Inverse transform deltas instead of raw state vectors
        position_unscaled = self.delta_position_scaler.inverse_transform(delta_position)
        velocity_unscaled = self.delta_velocity_scaler.inverse_transform(delta_velocity)
        return np.hstack([position_unscaled, velocity_unscaled])

    def get_raw_ground_truth(self, sat_id, offset, input_len, steps):
        samples_per_sat = 270
        start = sat_id * samples_per_sat + offset + input_len
        end = start + steps

        raw = self.raw_df[['position_x', 'position_y', 'position_z',
                       'velocity_x', 'velocity_y', 'velocity_z']].values
        return raw[start:end]



'''
#Sanity check
batch = next(iter(dataloader))
print("src:   ", batch['src'].shape)    # [BATCH_SIZE, 10, 7]
print("tgt:   ", batch['tgt'].shape)    # [BATCH_SIZE, 5, 7]
print("tgt_y: ", batch['tgt_y'].shape)  # [BATCH_SIZE, 5, 6]
'''

#MODEL TRAINING
if __name__ == "__main__":
    #Select Device
    device = torch.device('cuda')

    #Load Dataset
    train_dataset = OrbitDataset("new_training_data.csv", split ='train')
    val_dataset = OrbitDataset("new_training_data.csv", split ='val')

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4/10/2025 CHANGE 001
    delta_t = train_dataset.normalized_time_step

    #Instantiate model
    MODEL = OrbitAI(
        input_dim = INPUT_DIM,
        embed_dim = EMBED_DIM,
        output_dim = OUTPUT_DIM,
        num_heads = NUM_HEADS,
        feedforward_dim = FEED_FORWARD_DIM,
        num_layers = NUM_LAYERS,
        dropout = DROPOUT,
        seq_len = SEQ_LENGTH,
        pred_len = PRED_LEN
    ).to(device)

    #Select loss function
    CRITERION = nn.MSELoss()
    #Select optimizer
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    #Early stopping to prevent overfitting
    best_val_loss = float('inf')
    patience = 10               #Number of epochs to wait
    counter = 0                 #Tracking how many epochs we've waited
    early_stop = False          #Break this to stop

    #Training Loop
    for epoch in range(EPOCHS):
        start_time = time.time()

        sampling_ratio = min(0.5,epoch/50) #Gradually ramp up to 50%

        train_positions, train_velocities = train(MODEL, train_loader, OPTIMIZER, CRITERION, device, scheduled_sampling_ratio=sampling_ratio)
        validate_positions,validate_velocities = evaluate(MODEL, val_loader, CRITERION, device)

        train_loss = train_positions + train_velocities
        val_loss = validate_positions + validate_velocities

        elapsed = time.time() - start_time
        print(f"Epoch: {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.6f} (Pos: {train_positions:.6f}, Vel: {train_velocities:.6f}) | "
              f"Val Loss: {val_loss:.6f} (Pos: {validate_positions:.6f}, Vel: {validate_velocities:.6f}) | "
              f"Time: {elapsed:.2f}s")

        #Early Stopping
        if val_loss < best_val_loss:
            counter = 0
            torch.save(MODEL.state_dict(), 'optimal_weights_orbitai.pth')
            print("Validation loss improved. Model saved")
            best_val_loss = val_loss
        else:
            counter +=1
            print(f"No improvement in validation loss for {counter} epochs")

            if counter >= patience:
                print(f"Early stopping triggered at {epoch+1}")
                early_stop = True
                print(f"Best validation loss was: {best_val_loss:.6f}")
                break
