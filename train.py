import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.vla_clip_bc import VLAClipBC

def parse_args():
    p = argparse.ArgumentParser("Train VLA BC on MetaWorld npz dataset")
    p.add_argument("--data", type=str, required=True, help="Path to .npz from your collector")
    p.add_argument("--out", type=str, default="checkpoints/metaworld_soccer.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=512)

    # model knobs
    p.add_argument("--freeze-vision", action="store_true", help="Freeze CLIP vision (recommended at start)")
    p.set_defaults(freeze_vision=True)

    p.add_argument("--no-action-squash", dest="action_squash", action="store_false")
    p.set_defaults(action_squash=True)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class NPZBCDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.images = data["images"]   # (N,H,W,3) uint8
        self.states = data["states"]   # (N,S) float32
        self.actions = data["actions"] # (N,A) float32
        self.text_ids = data["text_ids"]  # (N,T) int64

        # metadata
        self.vocab = data["vocab"].item() if isinstance(data["vocab"], np.ndarray) else data["vocab"]
        self.env_name = str(data.get("env_name", ""))
        self.camera_name = str(data.get("camera_name", "topview"))
        self.flip_ud = bool(data.get("flip_ud", False))
        self.resize_to = int(data.get("resize_to", 480))
        self.instruction = str(data.get("instruction", ""))

        assert self.images.shape[0] == self.states.shape[0] == self.actions.shape[0] == self.text_ids.shape[0]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # uint8 (H,W,3)
        # to float tensor (3,H,W) in [0,1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        state_t = torch.from_numpy(self.states[idx]).float()
        action_t = torch.from_numpy(self.actions[idx]).float()
        text_t = torch.from_numpy(self.text_ids[idx]).long()

        return img_t, text_t, state_t, action_t


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bc_mse_loss(pred_action, gt_action):
    return F.mse_loss(pred_action, gt_action)


def train_one_epoch(model, loader, opt, device, grad_clip=1.0):
    model.train()
    total = 0.0
    n = 0
    for img, text_ids, state, action in loader:
        img = img.to(device)
        text_ids = text_ids.to(device)
        state = state.to(device)
        action = action.to(device)

        pred = model(img, text_ids, state)
        loss = bc_mse_loss(pred, action)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def compute_state_stats(states_np: np.ndarray):
    mean = states_np.mean(axis=0).astype(np.float32)
    std = states_np.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ds = NPZBCDataset(args.data)
    state_dim = ds.states.shape[1]
    action_dim = ds.actions.shape[1]
    vocab = ds.vocab
    vocab_size = int(max(vocab.values())) + 1

    # dataset stats for state normalization
    state_mean, state_std = compute_state_stats(ds.states)
    batch_size = 20
    num_workers = 0
    
    # dataset = TrainingDataset(args.data, resize_to=480, flip_ud=True)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    
    # for img, state, action, text_ids in loader:
    #     print("Image batch shape:", img.shape)


    model = VLAClipBC(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        pad_id=0,
        freeze_vision=args.freeze_vision,
        action_squash=args.action_squash,
    ).to(device)

    # load state stats into model buffers
    model.state_mean.copy_(torch.from_numpy(state_mean).to(device))
    model.state_std.copy_(torch.from_numpy(state_std).to(device))

    # optimize only trainable params (vision frozen -> only text/state/fusion/head)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        avg = train_one_epoch(model, loader, opt, device, grad_clip=args.grad_clip)
        print(f"epoch {ep:03d} | loss {avg:.6f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "d_model": args.d_model,

        # match your test-time preprocessing expectations
        "env_name": ds.env_name,
        "camera_name": ds.camera_name,
        "flip_ud": bool(ds.flip_ud),
        "resize_to": int(ds.resize_to),

        # normalization for inference
        "state_mean": state_mean,
        "state_std": state_std,
    }
    torch.save(ckpt, args.out)
    print("Saved checkpoint:", args.out)
    print(" camera_name:", ds.camera_name, "flip_ud:", ds.flip_ud, "resize_to:", ds.resize_to)


if __name__ == "__main__":
    main()
