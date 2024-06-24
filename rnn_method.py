import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class SalesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SalesRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out


class RNNModel():
    
    def __init__(self):
        self.prepare()
        self.run()

    
    def prepare(self):
        self.determine_device()
        self.load_data()


    def run(self):
        self.normalize_data()
        self.create_tensors()
        self.create_tensor_data_set()
        self.set_params()
        self.set_model_instance()
        self.set_optimizers()
        self.initial_training()


    def determine_device(self):
        self.self.device = torch.self.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using: {self.self.device}')


    def load_data(self):
        self.data = pd.read_csv('model_data/result_Plovdiv.csv')


    def normalize_data(self):
        scaler = MinMaxScaler()
        self.data_normalized = scaler.fit_transform(self.data)


    def create_tensors(self):
        seq_length = 12
        x = []
        y = []
        for i in range(len(self.data_normalized) - seq_length):
            x.append(self.data_normalized[i:i + seq_length])
            y.append(self.data_normalized[i + seq_length])

            self.x = np.array(x)
            self.y = np.array(y)
            self.x_tensor = torch.tensor(x, dtype=torch.float32)
            self.y_tensor = torch.tensor(y, dtype=torch.float32)


    def create_tensor_data_set(self):
        batch_size = 2
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    def set_params(self):
        self.input_size = self.data.shape[1]
        self.hidden_size = 64
        self.output_size = self.input_size
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 24000
        self.save_checkpoint_every = 1000
        self.train_losses = []
        self.val_losses = []


    def set_model_instance(self):
        self.model = SalesRNN(self.input_size, self.hidden_size, self.output_size, self.num_layers).to(self.device)


    def set_optimizers(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)  


    def initial_training(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}')

            if (epoch + 1) % self.save_checkpoint_every == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss.item(),
                }, f'rnn_training_data/sales_rnn_checkpoint_epoch_{epoch+1}.pth')
                print(f'Checkpoint saved at epoch {epoch + 1}')


        initial_input = self.data_normalized[-self.seq_length:].reshape(1, self.seq_length, self.input_size)
        initial_input = torch.tensor(initial_input, dtype=torch.float32).to(self.device)


        checkpoint_path = 'rnn_training_data/sales_rnn_checkpoint_epoch_250.pth'
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()


    def continue_training(self):
        checkpoint = torch.load('sales_rnn.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']

        num_epochs_continue = 10000

        for epoch in range(num_epochs_continue):
            self.model.train()
            outputs = self.model(self.x_tensor)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, self.y_tensor)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
            
            self.model.eval()
            val_outputs = self.model(self.x)
            val_loss = self.criterion(val_outputs, self.y)
            val_losses.append(val_loss.item())
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs_continue}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'sales_rnn_continued.pth')


    def predict_sales(self):
        months_per_year = 12
        predicted_sales_normalized = []

        checkpoint_path = 'rnn_training_data/sales_rnn_checkpoint_epoch_83000.pth' 
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()


        initial_input = self.data_normalized[-self.seq_length:].reshape(1, self.seq_length, self.input_size)
        initial_input = torch.tensor(initial_input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            current_input = initial_input
            for _ in range(months_per_year):
                prediction = self.model(current_input)
                predicted_sales_normalized.append(prediction.cpu().numpy())
                prediction = prediction.reshape(1, 1, self.input_size).to(self.device)
                current_input = torch.cat((current_input[:, 1:, :], prediction), dim=1)

        predicted_sales_normalized = np.array(predicted_sales_normalized).squeeze()
        predicted_sales = self.scaler.inverse_transform(predicted_sales_normalized)
        predicted_sales = np.rint(predicted_sales).astype(int)
        predicted_sales_df = pd.DataFrame(predicted_sales, columns=self.data.columns)
        predicted_sales_df.to_csv('predicted_sales_next_year_Sofia.csv', index=False)
