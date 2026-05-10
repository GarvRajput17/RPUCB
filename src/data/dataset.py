import os
import random
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class RecTrainDataset(Dataset):
    def __init__(self, train_pairs, user_train_items, num_items, interaction_rows, interaction_cols, num_negatives):
        self.train_pairs = train_pairs
        self.user_train_items = user_train_items
        self.num_items = num_items
        self.interaction_rows = interaction_rows
        self.interaction_cols = interaction_cols
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        u, i = self.train_pairs[idx]
        
        neg_items = []
        user_interacted = self.user_train_items[u]
        
        # Fast bulk sampling using numpy instead of while loop
        while len(neg_items) < self.num_negatives:
            candidates = np.random.randint(0, self.num_items, size=self.num_negatives * 2)
            for neg in candidates:
                neg = int(neg)
                if neg not in user_interacted:
                    neg_items.append(neg)
                    if len(neg_items) == self.num_negatives:
                        break
        
        return {
            'user': u,
            'pos_item': i,
            'neg_items': neg_items
        }


class RecDataset:
    def __init__(self, data_path, num_negatives=4, seed=None):
        self.data_path = data_path
        self.num_negatives = num_negatives
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_file = os.path.join(data_path, "train.rating")
        test_rating_file = os.path.join(data_path, "test.rating")
        test_negative_file = os.path.join(data_path, "test.negative")

        # 1. Read train.rating
        train_u = []
        train_i = []
        self.train_pairs = []
        self.user_train_items = defaultdict(set)
        
        with open(train_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    u, i = int(parts[0]), int(parts[1])
                    train_u.append(u)
                    train_i.append(i)
                    self.train_pairs.append((u, i))
                    self.user_train_items[u].add(i)

        # 2. Read test.rating to make sure num_users/num_items covers the test set as well
        self.test_ratings = []
        test_u = []
        test_i = []
        with open(test_rating_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    u, i = int(parts[0]), int(parts[1])
                    test_u.append(u)
                    test_i.append(i)
                    self.test_ratings.append((u, i))

        all_u = train_u + test_u
        all_i = train_i + test_i
        
        self.num_users = max(all_u) + 1 if all_u else 0
        self.num_items = max(all_i) + 1 if all_i else 0

        # Create localized train matrix
        self.train_matrix = sp.lil_matrix((self.num_users, self.num_items), dtype=np.float32)
        for u, i in self.train_pairs:
            self.train_matrix[u, i] = 1.0

        # Normalizations
        train_csr = self.train_matrix.tocsr()
        
        # Calculate N_u (user_interaction_counts)
        counts = np.array(train_csr.sum(axis=1)).flatten()
        self.user_interaction_counts = torch.LongTensor(counts)

        # Calculate N_i (item_interaction_counts)
        item_counts = np.array(train_csr.sum(axis=0)).flatten()
        self.item_interaction_counts = torch.LongTensor(item_counts)

        row_sums = np.maximum(1.0, counts)
        col_sums = np.maximum(1.0, np.array(train_csr.sum(axis=0)).flatten())

        # Create user interaction rows
        row_normalized = train_csr.copy()
        for u in range(self.num_users):
            start = row_normalized.indptr[u]
            end = row_normalized.indptr[u+1]
            row_normalized.data[start:end] /= row_sums[u]
        self.interaction_rows = torch.from_numpy(row_normalized.toarray()).float()

        # Create item interaction columns
        col_normalized = self.train_matrix.tocsc()
        for i in range(self.num_items):
            start = col_normalized.indptr[i]
            end = col_normalized.indptr[i+1]
            col_normalized.data[start:end] /= col_sums[i]
            
        self.interaction_cols = torch.from_numpy(col_normalized.transpose().toarray()).float()

        # Load test_negatives
        self.test_negatives = [[] for _ in range(self.num_users)]
        with open(test_negative_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    user_item_str = parts[0].strip("()")
                    u = int(user_item_str.split(",")[0])
                    negs = [int(x) for x in parts[1:]]
                    self.test_negatives[u] = negs

    def get_train_dataloader(self, batch_size, shuffle=True):
        dataset = RecTrainDataset(
            self.train_pairs, 
            self.user_train_items, 
            self.num_items, 
            self.interaction_rows, 
            self.interaction_cols, 
            self.num_negatives
        )
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True
        )

    def get_test_data(self):
        return self.test_ratings, self.test_negatives
