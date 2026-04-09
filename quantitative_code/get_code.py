# file: residual_vq_vqvae.py
import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Config
# ----------------------------
CONFIG = {
    "in_dim": 2000,
    "latent_dim": 2048,   # encoder -> latent space dim 
    "R": 32,              
    "K": 1000,            
    "commitment_cost": 0.25,
    "ema_decay": 0.99,
    "eps": 1e-5,
    "batch_size": 512,
    "lr": 1e-3,
    "num_epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./vq_model_FB15K237"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ----------------------------
# Dataset wrapper for embeddings.npy
# ----------------------------
class EmbeddingDataset(Dataset): # 获取实体embedding
    def __init__(self, emb_np):
        # emb_np: numpy array (N, in_dim)
        self.emb = emb_np.astype(np.float32)
    def __len__(self): return self.emb.shape[0]
    def __getitem__(self, idx):
        return self.emb[idx]

# ----------------------------
# Encoder / Decoder
# ----------------------------
class Encoder(nn.Module): # 定义一个简单的线性层，拓展实体embedding的维度，以方便对应一个码本序列
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
    def forward(self, x):
        return self.net(x)   # (B, latent_dim)

class Decoder(nn.Module): # 定义一个解码层，方便将实体重构到原始维度，（从而可以继续计算KGE损失，以保持实体的结构化语义信息）
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim),
        )
    def forward(self, z):
        return self.net(z)

# ----------------------------
# Residual VQ module with EMA codebook update 残差量化及更新
# ----------------------------
class ResidualVQ(nn.Module):
    def __init__(self, R, K, D, ema_decay=0.99, eps=1e-5, device='cpu'):
        """
        R: number of residual stages  共有多少个残差阶段，就是共有多少个码本
        K: codebook size per stage  每阶段码本的大小
        D: codeword dimensionality  每个码本中的token维度
        """
        super().__init__()
        self.R = R
        self.K = K
        self.D = D
        self.device = device

        # codebooks: one parameter tensor per stage, shape (R, K, D)
        # we keep codebooks as buffers and use EMA updates (so not direct gradient-based)
        codebook = torch.randn(R, K, D) * 0.01
        self.register_buffer("codebook", codebook)  # (R, K, D)

        # EMA statistics
        self.register_buffer("ema_count", torch.zeros(R, K) + eps)  # avoid divide-by-zero
        self.register_buffer("ema_sum", torch.zeros(R, K, D))
        self.ema_decay = ema_decay
        self.eps = eps

    def forward(self, z):
        """
        z: (B, D) encoder outputs
        returns:
            z_q: (B, D) reconstructed quantized vector (sum of selected codewords)
            all_indices: (B, R) indices per stage
            commitment_loss: scalar
        """
        B = z.shape[0]
        device = z.device
        res = z  # residual
        selected_idxs = []
        selected_codewords = []

        # For each residual stage, pick nearest codeword to current residual
        for t in range(self.R):
            # codebook_t: (K, D)
            cb_t = self.codebook[t]  # buffer on same device as module
            # ensure same device
            if cb_t.device != device:
                cb_t = cb_t.to(device)
            # compute squared distances between res and each codeword
            # res: (B, D) -> (B, 1, D); cb_t: (K, D) -> (1, K, D)
            dist = torch.sum((res.unsqueeze(1) - cb_t.unsqueeze(0)) ** 2, dim=2)  # (B, K)
            idx = torch.argmin(dist, dim=1)  # (B,)
            selected_idxs.append(idx)
            # gather codewords
            code_t = cb_t[idx]  # (B, D)
            selected_codewords.append(code_t)
            # update residual
            res = res - code_t

        # sum selected codewords -> quantized representation
        z_q = torch.stack(selected_codewords, dim=0).sum(dim=0)  # (B, D)
        # commitment loss: encourage encoder output to be close to quantized
        commitment_loss = F.mse_loss(z, z_q.detach())

        # For backprop: use straight-through estimator: pass gradients to encoder
        z_q_st = z_q + (z - z_q).detach()

        indices = torch.stack(selected_idxs, dim=1)  # (B, R)
        return z_q_st, indices, commitment_loss

    @torch.no_grad()
    def ema_update(self, z, indices):
        """
        z: (B, D) original encoder outputs
        indices: (B, R) selected indices for each stage
        Update self.ema_count and self.ema_sum and derive new codebook via ema_sum / ema_count.
        Note: For residual VQ, the relevant vector to accumulate for stage t is the residual BEFORE subtraction by current codeword.
        We'll reconstruct residuals stage-by-stage to find the vectors assigned to each codebook.
        """
        device = self.codebook.device
        B = z.shape[0]
        z = z.detach().to(device)

        # reconstruct stage-wise residuals (same sequence as forward)
        res = z
        for t in range(self.R):
            idx_t = indices[:, t].to(device)  # (B,)
            # for EMA we accumulate the current residual itself assigned to idx_t
            # gather one-hot
            # accumulate count and sum
            # convert idx_t to one-hot mat
            one_hot = F.one_hot(idx_t, num_classes=self.K).type_as(self.ema_count)  # (B, K)
            # update counts: sum over batch
            count_update = one_hot.sum(dim=0)  # (K,)
            sum_update = (one_hot.unsqueeze(2) * res.unsqueeze(1)).sum(dim=0)  # (K, D)
            # EMA update
            self.ema_count[t] = self.ema_count[t] * self.ema_decay + count_update * (1 - self.ema_decay)
            self.ema_sum[t] = self.ema_sum[t] * self.ema_decay + sum_update * (1 - self.ema_decay)
            # compute new codebook entry
            n = self.ema_count[t].unsqueeze(1).clamp(min=self.eps)  # (K,1)
            new_cb_t = self.ema_sum[t] / n  # (K, D)
            self.codebook[t] = new_cb_t
            # subtract chosen codeword to move to next residual
            # get current codewords for batch
            cb_t = self.codebook[t].to(device)
            code_t = cb_t[idx_t]  # (B, D)
            res = res - code_t

    def save_codebooks(self, path):
        # saves codebooks as numpy array (R*K, D) or (R, K, D)
        np.save(path, self.codebook.cpu().numpy())

    def load_codebooks(self, path):
        data = np.load(path)
        data = torch.from_numpy(data).to(self.codebook.device)
        assert data.shape == self.codebook.shape
        self.codebook.copy_(data)

# ----------------------------
# Full VQ-VAE model wrapper VQ-VAE流程，先通过encoder给实体升维，再通过残差量化过程计算每个实体的离散token序列（主要是码本重构损失），再降维回原始维度
# ----------------------------
class VQVAE_Residual(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(cfg["in_dim"], cfg["latent_dim"])
        self.dec = Decoder(cfg["latent_dim"], cfg["in_dim"])
        self.quant = ResidualVQ(cfg["R"], cfg["K"], cfg["latent_dim"],
                                ema_decay=cfg["ema_decay"], eps=cfg["eps"], device=cfg["device"])

    def forward(self, x):
        z = self.enc(x)
        z_q, idxs, commit_loss = self.quant(z)
        x_recon = self.dec(z_q)
        return x_recon, z, z_q, idxs, commit_loss

# ----------------------------
# Training loop 定义完整的训练逻辑
# ----------------------------
def train(embeddings_np, cfg):
    device = cfg["device"]
    dataset = EmbeddingDataset(embeddings_np) # 数据准备
    dl = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False, num_workers=2)

    model = VQVAE_Residual(cfg).to(device) # 初始化模型与优化器
    opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec.parameters()), lr=cfg["lr"])
    # note: codebook updates use EMA and are updated manually

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_rec_loss = 0.0
        total_commit = 0.0
        n = 0
        for batch in tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}"):
            x = batch.to(device)
            opt.zero_grad()
            x_recon, z, z_q, idxs, commit_loss = model(x) # 分别是解码后的实体embedding、编码后的实体嵌入、实体量化嵌入、每阶段切片大小、码本构建损失
            rec_loss = F.mse_loss(x_recon, x    ) # 保障解码后的实体嵌入不偏离原来太多
            loss = rec_loss + cfg["commitment_cost"] * commit_loss
            loss.backward()
            opt.step()
            # EMA codebook update (on CPU-buffered codebooks inside quant)
            with torch.no_grad():
                model.quant.ema_update(z.detach().cpu(), idxs.detach().cpu())
            total_rec_loss += rec_loss.item() * x.size(0)
            total_commit += commit_loss.item() * x.size(0)
            n += x.size(0)

        print(f"Epoch {epoch+1}: rec_mse={total_rec_loss/n:.6e}, commit_mse={total_commit/n:.6e}")
        # optionally save checkpoints
        torch.save({
            "enc": model.enc.state_dict(),
            "dec": model.dec.state_dict(),
        }, os.path.join(cfg["save_dir"], f"vq_epoch_{epoch+1}.pt"))

    # 训练完成后，对所有向量重新编码
    # after training, compute indices for all embeddings (encode full dataset)
    model.eval()
    all_indices = []
    with torch.no_grad():
        for i in range(0, embeddings_np.shape[0], cfg["batch_size"]):
            batch = torch.from_numpy(embeddings_np[i:i+cfg["batch_size"]]).to(device)
            z = model.enc(batch)  # (B, D)
            # forward through quant but avoid STE logic; reuse quant forward to get indices
            _, idxs, _ = model.quant(z)
            all_indices.append(idxs.cpu().numpy())
    all_indices = np.vstack(all_indices)  # (N, R)
    # save codebooks and indices 保存码本
    model.quant.save_codebooks(os.path.join(cfg["save_dir"], "codebooks.npy"))
    np.save(os.path.join(cfg["save_dir"], "indices.npy"), all_indices)
    print("Saved codebooks and indices to", cfg["save_dir"])
    return model, all_indices

# ----------------------------
# Evaluation helpers
# ----------------------------
def reconstruct_from_indices(indices_np, codebooks_np):
    # indices_np: (N, R) ; codebooks_np: (R, K, D)
    R, K, D = codebooks_np.shape
    N = indices_np.shape[0]
    recon = np.zeros((N, D), dtype=np.float32)
    for t in range(R):
        recon += codebooks_np[t][indices_np[:, t]]
    return recon  # (N, D)

def compute_reconstruction_mse(embeddings_np, recon_np):
    return np.mean((embeddings_np - recon_np[:, :embeddings_np.shape[1]])**2)  # note: decoder mapping may alter dim

# ----------------------------
# Example usage (main)
# ----------------------------
if __name__ == "__main__":
    # load embeddings
    emb_path = "/home/.../SS4LLM/dataset/RotatE_FB15k237/entity_embedding.npy"  # user-provided
    embeddings = np.load(emb_path)  # shape (N, 1024)
    print(embeddings.shape)
    assert embeddings.shape[1] == CONFIG["in_dim"]

    # train
    model, indices = train(embeddings, CONFIG)

    # optional: reconstruct quantized latent and decode to reconstruct original space
    codebooks = np.load(os.path.join(CONFIG["save_dir"], "codebooks.npy"))  # shape (R,K,D)
    recon_latents = reconstruct_from_indices(indices, codebooks)  # (N, D)
    # if you want to decode latents to original dimension: use model.dec on torch tensor
    with torch.no_grad():
        recon_decoded = model.dec(torch.from_numpy(recon_latents).to(CONFIG["device"])).cpu().numpy()
    mse = compute_reconstruction_mse(embeddings, recon_decoded)
    print("Final reconstruction MSE:", mse) # 训练之后
