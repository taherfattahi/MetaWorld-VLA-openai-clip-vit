import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

from envs.metaworld_env import MetaWorldMT1Wrapper
from utils.tokenizer import SimpleTokenizer
from models.vla_clip_bc import VLAClipBC

def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Meta-World MT1")

    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/metaworld_soccer.pt")

    parser.add_argument("--env-name", type=str, default="soccer-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)

    parser.add_argument("--instruction", type=str, default="Shoot the ball into the goal")

    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")

    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos")

    parser.add_argument("--override-camera-name", type=str, default="",
                        help="If non-empty, use this camera_name instead of the checkpoint value.")
    parser.add_argument("--override-flip-ud", type=int, default=-1,
                        help="Set 1 to force flip_ud, 0 to force no flip, -1 to use checkpoint.")
    parser.add_argument("--override-resize-to", type=int, default=0,
                        help="If >0, force resize_to instead of checkpoint.")
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = ckpt["vocab"]
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    d_model = int(ckpt["d_model"])

    camera_name = ckpt.get("camera_name", "topview")
    flip_ud = bool(ckpt.get("flip_ud", False))
    resize_to = int(ckpt.get("resize_to", 64))

    vocab_size = max(vocab.values()) + 1
    tokenizer = SimpleTokenizer(vocab=vocab)

    model = VLAClipBC(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        pad_id=0,
        freeze_vision=True,      # vision is typically frozen in BC baseline
        action_squash=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # load state normalization if present
    if "state_mean" in ckpt and "state_std" in ckpt:
        model.state_mean.copy_(torch.tensor(ckpt["state_mean"], device=device))
        model.state_std.copy_(torch.tensor(ckpt["state_std"], device=device))

    return model, tokenizer, camera_name, flip_ud, resize_to

def preprocess_img(img_rgb_uint8: np.ndarray, flip_ud: bool, resize_to: int) -> torch.Tensor:
    """
    Returns float tensor (1,3,H,W) in [0,1], after flip+resize to match training.
    """
    x = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0  # (3,H,W)

    if flip_ud:
        x = torch.flip(x, dims=[1])

    if resize_to and (x.shape[1] != resize_to or x.shape[2] != resize_to):
        x = F.interpolate(
            x.unsqueeze(0),
            size=(resize_to, resize_to),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

    return x.unsqueeze(0)  # (1,3,H,W)

def torch_img_to_uint8_rgb(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: torch tensor with shape (1,3,H,W) or (3,H,W), range [0,1] (float)
    returns: uint8 RGB image (H,W,3)
    """
    if img_t.ndim == 4:
        img_t = img_t[0]
    img_t = img_t.detach().cpu().clamp(0, 1)
    img = (img_t * 255.0).byte().permute(1, 2, 0).numpy()  # (H,W,3) uint8 RGB
    return img

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[test] Loading checkpoint from {args.checkpoint}")
    model, tokenizer, ckpt_camera, ckpt_flip_ud, ckpt_resize_to = load_model_and_tokenizer(args.checkpoint, device)

    camera_name = args.override_camera_name if args.override_camera_name.strip() else ckpt_camera
    flip_ud = ckpt_flip_ud if args.override_flip_ud == -1 else bool(args.override_flip_ud)
    resize_to = ckpt_resize_to if args.override_resize_to <= 0 else int(args.override_resize_to)

    print("[test] Using camera_name:", camera_name)
    print("[test] Using flip_ud:", flip_ud)
    print("[test] Using resize_to:", resize_to)

    instr_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1,T)

    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=camera_name,
    )

    print(f"[test] Meta-World MT1 env: {args.env_name}")
    print(f"[test] state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    for ep in range(args.episodes):
        img, state, info = env.reset()
        step = 0
        ep_reward = 0.0
        frames = []

        done = False
        while not done and step < args.max_steps:
            img_t = preprocess_img(img, flip_ud=flip_ud, resize_to=resize_to).to(device)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # For visualization, preprocess to a larger size
            img_d = preprocess_img(img, flip_ud=flip_ud, resize_to=840).to(device)
            img_disp = torch_img_to_uint8_rgb(img_d)     # (H,W,3) uint8 RGB

            with torch.no_grad():
                action_t = model.act(img_t, text_ids, state_t)  # (1, action_dim)
            action_np = action_t.squeeze(0).cpu().numpy()

            img, state, reward, done, info = env.step(action_np)
            ep_reward += float(reward)
            step += 1
            frames.append(img_disp.copy())

        print(f"[test] Episode {ep+1}/{args.episodes}: reward={ep_reward:.3f}, steps={step}")

        if args.save_video:
            video_path = os.path.join(args.video_dir, f"{args.env_name}_{camera_name}_ep{ep+1:03d}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {video_path}")

    env.close()
    print("[test] Done.")


if __name__ == "__main__":
    main()
