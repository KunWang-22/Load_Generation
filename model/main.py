import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import argparse
import pickle
import os
import random

seed = 22
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



class Diffusion():
    def __init__(self, noise_step, beta_start, beta_end, data_length, device):
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.data_length = data_length
        self.device = device
        # 定义beta和对应的alpha
        self.beta = self.noise_schedule().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_step)
        return betas

    def forward_process(self, x_0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(-1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t]).unsqueeze(-1)
        #重参数技巧，通过正态分布采样然后变换得到加噪后的分布
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise
        return x_t, noise
    
    def timestep_sample(self, n):
        # 采样n个timestep用于训练模型，采用均匀分布
        time_steps = torch.randint(1, self.noise_step, (n,1))
        return time_steps

    def sample(self, model, n, condition):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.data_length)).unsqueeze(-1).to(self.device)
            for i in reversed(range(1, self.noise_step)):
                t = (torch.ones(n)*i).long().unsqueeze(-1).to(self.device)
                # conditional sample
                predicted_noise = model(x, t, condition)
                alpha_t = self.alpha[t].unsqueeze(-1)
                alpha_hat_t = self.alpha_hat[t].unsqueeze(-1)
                # beta此处对应方差，也可以用beta_tilde表示
                beta_t = self.beta[t].unsqueeze(-1)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha_t) * (x - (1-alpha_t)/torch.sqrt(1-alpha_hat_t) * predicted_noise) + torch.sqrt(beta_t) * noise
        model.train()
        # 此处可以根据数据情况，考虑是否添加数据归一化/反归一化的操作
        return x



class Timestep_Embedding(nn.Module):
    def __init__(self, embedding_dim, device):
        super(Timestep_Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.fc_1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc_2 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.SiLU()

    def forward(self, t):
        x = self._embedding(t)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        x = self.activation(x)
        return x
    
    def _embedding(self, t):
        t_seq = t.repeat(1, self.embedding_dim//2).to(self.device)
        frequency = torch.pow(10, torch.arange(self.embedding_dim//2) / (self.embedding_dim//2-1) * 4.0).to(self.device)
        emb_sin = torch.sin(t_seq * frequency)
        emb_cos = torch.cos(t_seq * frequency)
        embedding = torch.cat([emb_sin, emb_cos], dim=1)
        return embedding



class PE_Embedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, device):
        super(PE_Embedding, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(self.input_dim, self.embedding_dim)

    def forward(self, x):
        data_length = x.shape[1]
        x = self.fc(x)
        x = x + self._position_encoding(data_length)
        return x

    def _position_encoding(self, data_length):
        encoding = torch.zeros((data_length, self.embedding_dim)).to(self.device)
        position = torch.arange(data_length).unsqueeze(1).to(self.device)
        encoding[:, 0::2] = torch.sin( position / torch.pow(10000, torch.arange(0, self.embedding_dim, 2).to(self.device)/self.embedding_dim) )
        encoding[:, 1::2] = torch.cos( position / torch.pow(10000, torch.arange(1, self.embedding_dim, 2).to(self.device)/self.embedding_dim) )
        return encoding




class Encoder_Layer(nn.Module):
    def __init__(self, embedding_dim, num_head, dropout):
        super(Encoder_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, dropout=dropout)
        self.fc_1 = nn.Linear(self.embedding_dim, self.embedding_dim//2)
        self.fc_2 = nn.Linear(self.embedding_dim//2, self.embedding_dim)
        self.activation = nn.ReLU()
        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)

    def forward(self, x):
        residual = x
        x, _ = self.attention(query=x, key=x, value=x)
        x = self.layer_norm_1(x+residual)
        residual = x
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        x = self.layer_norm_2(x+residual)
        return x



class Condition_Embedding(nn.Module):
    def __init__(self, condition_input_dim, embedding_dim, num_head, num_layer, dropout, device):
        super(Condition_Embedding, self).__init__()
        self.pe_embedding = PE_Embedding(condition_input_dim, embedding_dim, device)
        self.encoder = nn.ModuleList([Encoder_Layer(embedding_dim, num_head, dropout) for _ in range(num_layer)])

    def forward(self, x):
        x = self.pe_embedding(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        return x



class Residual_Block(nn.Module):
    def __init__(self, condition_input_dim, embedding_dim, num_head, num_layer, dropout, device):
        super(Residual_Block, self).__init__()
        self.fc_timestep = nn.Linear(embedding_dim, embedding_dim)
        self.fc_condition = nn.Linear(embedding_dim, embedding_dim)
        self.fc_output = nn.Linear(embedding_dim//2, embedding_dim)
        self.attention = nn.ModuleList([Encoder_Layer(embedding_dim, num_head, dropout) for _ in range(num_layer)])
        self.condtion_embbeding = Condition_Embedding(condition_input_dim, embedding_dim, num_head, num_layer, dropout, device)

    def forward(self, x, timestep_emb, condition):
        residual = x
        x = x + self.fc_timestep(timestep_emb.unsqueeze(1))
        for attention_layer in self.attention:
            x = attention_layer(x)
        
        condition_emb = self.condtion_embbeding(condition)
        x = x + self.fc_condition(condition_emb)
        
        x_1, x_2 = torch.chunk(x, 2, dim=-1)
        x = torch.sigmoid(x_1) * torch.tanh(x_2)
        x = self.fc_output(x)
        residual = residual + x

        return residual, x



class Model(nn.Module):
    def __init__(self, input_dim, condition_input_dim, embedding_dim, num_head, num_layer, num_block, dropout, device):
        super(Model, self).__init__()
        self.device = device
        self.pe_embedding = PE_Embedding(input_dim, embedding_dim, device)
        self.timestep_embedding = Timestep_Embedding(embedding_dim, device)
        self.residual_model = nn.ModuleList([Residual_Block(condition_input_dim, embedding_dim, num_head, num_layer, dropout, device) for _ in range(num_block)])
        # check dim !!!
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        self.fc_concat = nn.Linear(embedding_dim, embedding_dim)
        self.fc_output = nn.Linear(embedding_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x, timestep, conditon):
        timestep_emb = self.timestep_embedding(timestep)
        
        x = self.pe_embedding(x)
        x = self.fc_input(x)
        x = self.activation(x)

        skip = []
        for residual_layer in self.residual_model:
            x, skip_output = residual_layer(x, timestep_emb, conditon)
            skip.append(skip_output)

        x = torch.sum(torch.stack(skip), dim=0)
        x = self.fc_concat(x)
        x = self.activation(x)
        x = self.fc_output(x)

        return x        



class Dataset_UKDA(Dataset):
    def __init__(self, file_path, aggregation_num, mode, test_user, test_day):
        origin, condition, scaler = get_dataset(file_path, aggregation_num, mode, test_user, test_day)
        self.scaler = scaler
        self.origin = torch.from_numpy(origin).type(torch.float32).unsqueeze(-1)
        self.condition = torch.from_numpy(condition).type(torch.float32).unsqueeze(-1)

    def __getitem__(self, index):
        return self.origin[index], self.condition[index]
    
    def __len__(self):
        return self.origin.shape[0]



def get_dataset(file_path, aggregation_num, mode, test_user, test_day):
    original_data = pd.read_csv(file_path)

    aggregated_data = pd.DataFrame()
    aggregated_data["time"] = pd.to_datetime(original_data["time"])
    for i in range((original_data.shape[1]-2)//aggregation_num):
        temp_data = original_data.iloc[:, (1+i*aggregation_num):(1+(i+1)*aggregation_num)].sum(axis=1)
        temp_name = "user_" + str(i+1)
        aggregated_data[temp_name] = temp_data

    scaler = StandardScaler().fit(aggregated_data.iloc[:, 1:])
    aggregated_data.iloc[:, 1:] = scaler.transform(aggregated_data.iloc[:, 1:])

    aggregated_data["month"] = [aggregated_data["time"][i].month for i in range(aggregated_data.shape[0])]
    aggregated_data["day"] = [aggregated_data["time"][i].day for i in range(aggregated_data.shape[0])]
    aggregated_data["hour"] = [aggregated_data["time"][i].hour for i in range(aggregated_data.shape[0])]
    aggregated_data["minute"] = [aggregated_data["time"][i].minute for i in range(aggregated_data.shape[0])]
    month_index = aggregated_data["month"].value_counts(sort=False)//48

    condition_data = pd.DataFrame()
    condition_data["time"] = aggregated_data["time"]
    condition_df = aggregated_data.groupby(["month", "day", "hour", "minute"]).mean(numeric_only=False).round(3)
    for user in aggregated_data.columns[1:-4]:
        user_data = np.array([])
        for month in month_index.index:
            days = month_index[month]
            temp_data = condition_df[user].loc[month, 14, :, :].values.reshape(1,-1)
            month_data = temp_data.repeat(days, axis=0)
            user_data = np.append(user_data, month_data.flatten())
        condition_data[user] = user_data

    all_origin = aggregated_data.iloc[:, 1:-4].values.T.reshape(condition_data.shape[1]-1, -1, 48)
    all_condition = condition_data.iloc[:, 1:].values.T.reshape(condition_data.shape[1]-1, -1, 48)

    if mode == "train":
        origin = np.concatenate((all_origin[:-test_user].reshape(-1, 48), all_origin[-test_user:, :-test_day, :].reshape(-1, 48)), axis=0)
        condition = np.concatenate((all_condition[:-test_user].reshape(-1, 48), all_condition[-test_user:, :-test_day, :].reshape(-1, 48)), axis=0)
    elif mode == "test":
        origin = all_origin[-test_user:, -test_day:, :].reshape(-1,48)
        condition = all_condition[-test_user:, -test_day:, :].reshape(-1,48)
        # print(origin.shape)
    return origin, condition, scaler



def train(args, device):
    train_dataset = Dataset_UKDA(args.file_path, args.aggregation_num, args.mode, args.test_user, args.test_day)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    model = Model(args.input_dim, args.condition_input_dim, args.embedding_dim, args.num_head, args.num_layer, args.num_block, args.dropout, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 这次可以使用“MSE+KL"的指标共同作为模型的loss
    criterion = nn.MSELoss()
    diffusion = Diffusion(noise_step=args.noise_step, beta_start=args.beta_start, beta_end=args.beta_end, data_length=args.data_length, device=device)
    
    print("Start Traning !!!")
    for epoch in range(args.num_epoch):
        losses = []
        for i, (data, condition) in enumerate(train_dataloader):
            data = data.to(device)
            condition = condition.to(device)
            t = diffusion.timestep_sample(data.shape[0]).to(device)
            x_t, noise = diffusion.forward_process(data, t)
            predicted_noise = model(x_t, t, condition)

            loss = criterion(noise, predicted_noise)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), "../log/model/model.pt")
        print("Epoch {}/{}, Loss: {}".format(epoch+1, args.num_epoch, np.array(losses, dtype=float).mean()))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("The hyper-parameters of this project")
    # Dataset 参数
    parser.add_argument("--file_path", type=str, default="../data/UKDA_2013_clean.csv")
    parser.add_argument("--aggregation_num", type=int, default=10)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--test_user", type=int, default=10)
    parser.add_argument("--test_day", type=int, default=30)
    # Diffusion 参数
    parser.add_argument("--noise_step", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.5)
    parser.add_argument("--data_length", type=int, default=48)
    # Model 参数
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--condition_input_dim", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--num_layer", type=int, default=6)
    parser.add_argument("--num_block", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    # 特别注意device的构建
    args.cuda = not args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    train(args, device)