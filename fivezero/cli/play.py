"""Simple CLI to play FiveZero against a trained network."""

import argparse
import sys

import torch

from fivezero.gameEngine import Actor, N, legal_moves, new_game, render, step, terminal_value, winner
from fivezero.net import ConvNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play FiveZero against a saved network.")
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        help="Path to a torch checkpoint (state_dict or full model) for ConvNet.",
    )
    parser.add_argument(
        "--human",
        choices=["x", "o"],
        default="x",
        help="Choose your side: X plays first (Actor.POSITIVE), O plays second.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string, e.g. cpu or cuda (defaults to cpu).",
    )
    return parser.parse_args()


def load_network(model_path: str, device: torch.device) -> ConvNet:
    net = ConvNet(device)
    checkpoint = torch.load(model_path, map_location=device)
    try:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, torch.nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
        net.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load model from {model_path}") from exc
    net.eval()
    return net


def choose_network_move(net: ConvNet, state, print_net_outputs: bool = True) -> int:
    with torch.no_grad():
        logits = net.forward_policy(net.encode(state)).squeeze(0)
        values = net.forward_value(net.encode(state)).squeeze(0)

    if print_net_outputs:
        print(f"Logits: {logits}")
        print(f"Values: {values}")

    moves = torch.tensor(legal_moves(state), device=logits.device, dtype=torch.long)
    masked_logits = torch.full_like(logits, float("-inf"))
    masked_logits[moves] = logits[moves]
    return int(torch.argmax(masked_logits).item())


def prompt_human_move(state) -> int:
    moves = set(int(m) for m in legal_moves(state))
    prompt = "Enter your move as 'row col' (0-indexed) or a single index 0-24 (q to quit): "
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"q", "quit", "exit"}:
            print("Exiting game.")
            raise SystemExit(0)
        parts = raw.replace(",", " ").split()
        try:
            if len(parts) == 1:
                move = int(parts[0])
            elif len(parts) == 2:
                row = int(parts[0])
                col = int(parts[1])
                move = row * N + col
            else:
                raise ValueError
        except ValueError:
            print("Invalid format. Provide 'row col' or a single index.")
            continue

        if move not in moves:
            open_spots = ", ".join(f"({m // N},{m % N})" for m in sorted(moves))
            print(f"Illegal move. Open spots: {open_spots}")
            continue
        return move


def play_game(net: ConvNet, human_actor: Actor) -> None:
    state = new_game()
    print(f"You are {'X' if human_actor == Actor.POSITIVE else 'O'}.")
    print("X moves first. Enter Ctrl+C or 'q' to quit.\n")

    while True:
        print(render(state))
        term = terminal_value(state)
        if term is not None:
            break

        if state.player == human_actor:
            move = prompt_human_move(state)
        else:
            move = choose_network_move(net, state)
            r, c = divmod(move, N)
            print(f"Network plays: {r} {c}")

        state = step(state, move)
        print()

    print(render(state))
    result = winner(state.board)
    if result == 0:
        print("Game over: draw.")
    elif result == human_actor:
        print("Game over: you win!")
    else:
        print("Game over: network wins.")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    net = load_network(args.model_path, device)
    human_actor = Actor.POSITIVE if args.human == "x" else Actor.NEGATIVE

    try:
        play_game(net, human_actor)
    except KeyboardInterrupt:
        print("\nExiting game.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
