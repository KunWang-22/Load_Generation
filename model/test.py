import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def test(args, device):
    test_dataset = Dataset_UKDA(args.file_path, args.aggregation_num, args.mode, args.test_user, args.test_day)
    test_dataloader = DataLoader(test_dataset, len(test_dataset), shuffle=True)

    model = Model(args.input_dim, args.condition_input_dim, args.embedding_dim, args.num_head, args.num_layer, args.num_block, args.dropout, device).to(device)
    # model.load_state_dict(torch.load("../log/model/model.pt"))
    diffusion = Diffusion(noise_step=args.noise_step, beta_start=args.beta_start, beta_end=args.beta_end, data_length=args.data_length, device=device)

    print("Start Test !!!")
    # real_data = []
    # generated_data = []
    for i, (data, condition) in enumerate(test_dataloader):
        condition = condition.to(device)
        # print(condition.shape)
        predicted = diffusion.sample(model, len(test_dataloader), condition)

        reverse_real_data = reverse_data(data.squeeze(-1).reshape(args.test_user, -1), test_dataset.scaler.mean_[-args.test_user:].reshape(-1,1), test_dataset.scaler.mean_[-args.test_user:].reshape(-1,1))
        print(reverse_real_data.shape)
        reverse_generated_data = reverse_data(predicted.cpu().detach().squeeze(-1).reshape(args.test_user, -1), test_dataset.scaler.mean_[-args.test_user:].reshape(-1,1), test_dataset.scaler.mean_[-args.test_user:].reshape(-1,1))
        # print(reverse_real_data)
        # print(reverse_generated_data)
    np.save("../result/real_data.npy", reverse_real_data.reshape(args.test_user, args.test_day, -1).numpy())
    np.save("../result/generated_data.npy", reverse_generated_data.reshape(args.test_user, args.test_day, -1).numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser("The hyper-parameters of this project")
    # Dataset 参数
    parser.add_argument("--file_path", type=str, default="../data/UKDA_2013_clean.csv")
    parser.add_argument("--aggregation_num", type=int, default=10)
    parser.add_argument("--mode", type=str, default="test")
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

    test(args, device)