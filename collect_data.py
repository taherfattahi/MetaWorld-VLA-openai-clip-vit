import os
import argparse
import time
import numpy as np
import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP
from utils.tokenizer import SimpleTokenizer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type=str, default="soccer-v3")

    parser.add_argument(
        "--camera-name",
        type=str,
        default="corner",
        help="Meta-World camera: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)

    parser.add_argument("--output-path", type=str, default="data/metaworld_bc_corner_flipped.npz")

    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional sleep between steps for visualization (seconds)")

    parser.add_argument("--instruction", type=str, default="Shoot the ball into the goal",
                        help="Fixed instruction for all episodes")

    # Flip/resize options
    parser.add_argument("--no-flip-ud", dest="flip_ud", action="store_false",
                        help="Disable vertical flip (top<->bottom). Default is enabled.")
    parser.set_defaults(flip_ud=True)

    parser.add_argument("--resize-to", type=int, default=0,
                        help="If > 0, resize frames to (resize_to, resize_to) before saving")

    # Visualization options
    parser.add_argument("--vis", action="store_true", help="visualize frames")
    parser.add_argument("--vis-every", type=int, default=1, help="show every Nth sample")
    parser.add_argument("--vis-delay", type=int, default=1, help="cv2.waitKey delay ms")

    return parser.parse_args()


def extract_state(obs):
    """Meta-World MT1 observations are flat numpy arrays already."""
    return np.asarray(obs, dtype=np.float32).ravel()


def maybe_resize_np(img: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return img
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        # Torch fallback (no cv2 required)
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)  # (1,3,H,W)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        out = x.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    env = gym.make(
        "Meta-World/MT1",
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )

    obs, info = env.reset(seed=args.seed)
    policy = ENV_POLICY_MAP[args.env_name]()

    images, states, actions, texts = [], [], [], []
    instruction = args.instruction

    # Lazy import for visualization only
    cv2 = None
    if args.vis:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            raise RuntimeError("cv2 is required for --vis but is not installed.")

    sample_idx = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < args.max_steps:
            action = policy.get_action(obs)

            img = env.render()  # uint8 RGB (H,W,3)

            # Apply flip BEFORE saving (this defines your dataset "camera frame")
            if args.flip_ud:
                img = img[::-1, :, :].copy()

            # Optional resize BEFORE saving
            img = maybe_resize_np(img, args.resize_to)

            state = extract_state(obs)

            images.append(img)
            states.append(state.copy())
            actions.append(np.asarray(action, dtype=np.float32).copy())
            texts.append(instruction)

            # Visualization (optional)
            if args.vis and (sample_idx % max(1, args.vis_every) == 0):
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr, f"ep={ep+1} step={steps}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
                )
                cv2.imshow("MetaWorld (saved frame)", bgr)
                key = cv2.waitKey(args.vis_delay) & 0xFF
                if key == ord("q") or key == 27:
                    done = True
                    break

            # Step environment
            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1
            sample_idx += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep+1}/{args.episodes} finished after {steps} steps, success={int(info.get('success', 0))}")

    env.close()
    if cv2 is not None:
        cv2.destroyAllWindows()

    images = np.stack(images, axis=0)   # (N,H,W,3)
    states = np.stack(states, axis=0)   # (N,state_dim)
    actions = np.stack(actions, axis=0) # (N,action_dim)

    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,

        # metadata for consistency
        env_name=args.env_name,
        camera_name=args.camera_name,
        flip_ud=np.array(bool(args.flip_ud)),
        resize_to=np.array(int(args.resize_to)),
        instruction=args.instruction,
    )

    print("Saved dataset to", args.output_path)
    print("  images:", images.shape)
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  text_ids:", text_ids.shape)
    print("  camera_name:", args.camera_name, "flip_ud:", args.flip_ud, "resize_to:", args.resize_to)


if __name__ == "__main__":
    main()
