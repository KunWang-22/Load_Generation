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

from main import Dataset_UKDA, Model, Diffusion


def reverse_data(data, mean, std):
    reversed_data = data * std + mean
    return reversed_data


def test_new(args, device):
    all_data = pd.read_csv(args.file_path)
    
    test = all_data.iloc[:, -51:-1].sum(axis=1)

    aggregated_data = pd.DataFrame()
    aggregated_data["time"] = pd.to_datetime(all_data["time"])
    for i in range((all_data.shape[1]-2)//50):
        temp_data = all_data.iloc[:, (1+i*50):(1+(i+1)*50)].sum(axis=1)
        temp_name = "user_" + str(i+1)
        aggregated_data[temp_name] = temp_data

    scaler = StandardScaler().fit(aggregated_data.iloc[:, 1:])    
    #     # scaler = StandardScaler().fit(aggregated_data)
    new_test = (test - scaler.mean_[10]) / scaler.scale_[10]

    test_data = new_test[48*30*0:48*30*1].values.reshape(-1, 48)
    test_condition = test_data[14].reshape(1, -1).repeat(30, axis=0)

    model = Model(args.input_dim, args.condition_input_dim, args.embedding_dim, args.num_head, args.num_layer, args.num_block, args.dropout, device).to(device)
    model.load_state_dict(torch.load("../log/model/model.pt"))
    print("Load model successfully !!!")
    diffusion = Diffusion(noise_step=args.noise_step, beta_start=args.beta_start, beta_end=args.beta_end, data_length=args.data_length, device=device)

    print("Start Test Add !!!")

    data = torch.from_numpy(test_data).type(torch.float32).unsqueeze(-1)
    condition = torch.from_numpy(test_condition).type(torch.float32).unsqueeze(-1).to(device)
    predicted = diffusion.sample(model, 30, condition)

    np.save("../result/real_data_add.npy", data.numpy())
    np.save("../result/generated_data_add.npy", predicted.cpu().detach().numpy())


def test(args, device):
    test_dataset = Dataset_UKDA(args.file_path, args.aggregation_num, args.mode, args.test_user, args.test_day, args.validation_num)
    test_dataloader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

    model = Model(args.input_dim, args.condition_input_dim, args.embedding_dim, args.num_head, args.num_layer, args.num_block, args.dropout, device).to(device)
    model.load_state_dict(torch.load("../log/model/model.pt"))
    print("Load Model Successfully !!!")
    diffusion = Diffusion(noise_step=args.noise_step, beta_start=args.beta_start, beta_end=args.beta_end, data_length=args.data_length, device=device)

    print("Start Test !!!")
    repeat_type = "once"
    if args.mode == "test":
        for i, (data, condition) in enumerate(test_dataloader):
            condition = condition.to(device)
            predicted = diffusion.sample(model, len(test_dataset), condition)

            reverse_real_data = reverse_data(data.squeeze(-1).reshape(args.test_user, -1), test_dataset.scaler.mean_[-args.test_user:].reshape(-1,1), test_dataset.scaler.scale_[-args.test_user:].reshape(-1,1))
            reverse_generated_data = reverse_data(predicted.cpu().detach().squeeze(-1).reshape(args.test_user, -1), test_dataset.scaler.mean_[-args.test_user:].reshape(-1,1), test_dataset.scaler.scale_[-args.test_user:].reshape(-1,1))

            real_data = reverse_real_data.reshape(args.test_user, args.test_day, -1).numpy()
            generated_data = reverse_generated_data.reshape(args.test_user, args.test_day, -1).numpy()

        if args.repeat == 0:
            print("Test Once !!!")
        else:
            print("Test with Repeat !!!")
            repeat_type = "repeat"
            for _ in range(args.repeat-1):
                for i, (data, condition) in enumerate(test_dataloader):
                    condition = condition.to(device)
                    predicted = diffusion.sample(model, len(test_dataset), condition)

                    reverse_real_data = reverse_data(data.squeeze(-1).reshape(args.test_user, -1), test_dataset.scaler.mean_[-args.test_user-args.validation_num:-args.validation_num].reshape(-1,1), test_dataset.scaler.scale_[-args.test_user-args.validation_num:-args.validation_num].reshape(-1,1))
                    reverse_generated_data = reverse_data(predicted.cpu().detach().squeeze(-1).reshape(args.test_user, -1), test_dataset.scaler.mean_[-args.test_user-args.validation_num:-args.validation_num].reshape(-1,1), test_dataset.scaler.scale_[-args.test_user-args.validation_num:-args.validation_num].reshape(-1,1))

                    real_data_temp = reverse_real_data.reshape(args.test_user, args.test_day, -1).numpy()
                    generated_data_temp = reverse_generated_data.reshape(args.test_user, args.test_day, -1).numpy()

                    real_data = np.concatenate((real_data, real_data_temp), axis=-1)
                    generated_data = np.concatenate((generated_data, generated_data_temp), axis=-1)
            
            real_data.reshape(args.test_user, args.test_day, -1, 48)
            generated_data.reshape(args.test_user, args.test_day, -1, 48)
        
    elif args.mode == "validation":
        for i, (data, condition) in enumerate(test_dataloader):
            condition = condition.to(device)
            predicted = diffusion.sample(model, len(test_dataset), condition)

            reverse_real_data = reverse_data(data.squeeze(-1).reshape(args.validation_num, -1), test_dataset.scaler.mean_[-args.validation_num:].reshape(-1,1), test_dataset.scaler.scale_[-args.validation_num:].reshape(-1,1))
            reverse_generated_data = reverse_data(predicted.cpu().detach().squeeze(-1).reshape(args.validation_num, -1), test_dataset.scaler.mean_[-args.validation_num:].reshape(-1,1), test_dataset.scaler.scale_[-args.validation_num:].reshape(-1,1))

            real_data = reverse_real_data.reshape(args.validation_num, 365, -1).numpy()
            generated_data = reverse_generated_data.reshape(args.validation_num, 365, -1).numpy()

    np.save("../result/real_data_"+args.mode+"_"+repeat_type+".npy", real_data)
    np.save("../result/generated_data_"+args.mode+"_"+repeat_type+".npy", generated_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("The hyper-parameters of this project")
    # Dataset 参数
    parser.add_argument("--file_path", type=str, default="../data/UKDA_2013_clean.csv")
    parser.add_argument("--aggregation_num", type=int, default=10)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--test_user", type=int, default=10)
    parser.add_argument("--test_day", type=int, default=60)
    parser.add_argument("--validation_num", type=int, default=6)
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
    # Training 参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    # Test 参数
    parser.add_argument("--repeat", type=int, default=0)

    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    # 特别注意device的构建
    args.cuda = not args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    test(args, device)
    # test_new(args, device)