import os
import glob
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from diffusion_model import ConditionalUnet1D, MapEncoder

# --- Dataset 定义 ---
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, expected_horizon=100):
        self.raw_data = []
        files = glob.glob(os.path.join(data_dir, "*.pkl"))
        print(f"Loading {len(files)} data files...")
        
        for f_path in files:
            with open(f_path, 'rb') as f:
                batch_data = pickle.load(f)
                self.raw_data.extend(batch_data)
        
        # 清洗轨迹长度不一致的数据
        self.data = []
        rejected_count = 0
        
        for sample in self.raw_data:
            # 检查 trajectory 形状: 应该是 (HORIZON, 2)
            traj = sample['traj']
            if traj.shape[0] == expected_horizon and traj.shape[1] == 2:
                self.data.append(sample)
            else:
                rejected_count += 1
                
        print(f"Total raw samples: {len(self.raw_data)}")
        print(f"Valid samples: {len(self.data)} (Rejected {rejected_count} bad shapes)")
        
        if len(self.data) == 0:
            raise RuntimeError("No valid data found! Check HORIZON settings.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        map_tensor = torch.from_numpy(sample['map']).float()
        goal_tensor = torch.from_numpy(sample['goal']).float() / 10.0
        
        # 转置 (HORIZON, 2) -> (2, HORIZON)
        # 使用 .copy() 确保没有负步长问题
        traj_numpy = sample['traj'].astype(np.float32).copy()
        traj_tensor = torch.from_numpy(traj_numpy).float().transpose(0, 1) / 10.0
        
        return map_tensor, goal_tensor, traj_tensor

# --- Training Loop ---
def train():
    # 配置
    DATA_DIR = "./expert_data"
    OUTPUT_DIR = "./diffusion_weights"
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 数据加载
    dataset = TrajectoryDataset(DATA_DIR)
    if len(dataset) == 0:
        print("Error: No data found! Run collect_data.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. 模型初始化
    map_encoder = MapEncoder(input_channels=1, feature_dim=128).to(DEVICE)
    noise_net = ConditionalUnet1D(input_dim=2, cond_dim=128+2).to(DEVICE)
    
    optimizer = AdamW(list(map_encoder.parameters()) + list(noise_net.parameters()), lr=LR)
    
    # 3. 训练主循环
    print("Start Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        map_encoder.train()
        noise_net.train()
        
        for maps, goals, trajs in dataloader:
            maps = maps.to(DEVICE)
            goals = goals.to(DEVICE)
            trajs = trajs.to(DEVICE) # (B, 2, T) Ground Truth Trajectory (x_0)
            
            batch_size = maps.shape[0]
            
            # --- Diffusion Forward Process ---
            # 1. Sample timesteps t ~ Uniform(0, T)
            t = torch.randint(0, 50, (batch_size,), device=DEVICE).long() # 假设 50 步
            
            # 2. Create Noise epsilon
            noise = torch.randn_like(trajs)
            
            # 3. Add noise to trajectory (Forward SDE/DDPM)
            # 为了简化代码，这里手写一个简单的 linear beta schedule
            # 实际生产建议封装到类里，这里 hardcode 参数保持与 inference 一致
            betas = torch.linspace(0.0001, 0.02, 50).to(DEVICE)
            alphas = 1 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            # Extract alpha_bar_t
            sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None]
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t])[:, None, None]
            
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
            noisy_trajs = sqrt_alpha_bar * trajs + sqrt_one_minus_alpha_bar * noise
            
            # --- Model Prediction ---
            # Encode map
            map_feats = map_encoder(maps) # (B, 128)
            # Concat conditions
            global_cond = torch.cat([map_feats, goals], dim=1) # (B, 130)
            
            # Predict NOISE
            noise_pred = noise_net(noisy_trajs, t, global_cond)
            
            # --- Loss ---
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                'map_encoder': map_encoder.state_dict(),
                'noise_pred_net': noise_net.state_dict(),
            }, save_path)
            
            # 保存 latest
            torch.save({
                'map_encoder': map_encoder.state_dict(),
                'noise_pred_net': noise_net.state_dict(),
            }, os.path.join(OUTPUT_DIR, "ckpt_latest.pt"))

    print("Training finished!")

if __name__ == '__main__':
    train()
