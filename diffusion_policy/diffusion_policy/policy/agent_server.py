"""Serve a frozen base policy to the HiRe-MoMa actor (agentlace, request "predict").

  python -m diffusion_policy.policy.agent_server --ckpt ckpt/place_only/ep1000_ema.pt --port 5555
"""
import argparse

from agentlace.inference import InferenceServer

from diffusion_policy.policy.agent import DiffusionPolicyAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--act_horizon", type=int, default=6, help="Actions returned per query.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    agent = DiffusionPolicyAgent(args.ckpt, act_horizon=args.act_horizon, device=args.device)
    server = InferenceServer(port_num=args.port)
    server.register_interface("predict", lambda obs: {"actions": agent.predict_chunk(obs)})
    print(f"Base policy server ready on port {args.port}", flush=True)
    server.start(threaded=False)


if __name__ == "__main__":
    main()
