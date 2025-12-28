"""
Deploy an agent to generals.io live servers.

This example shows how to run an agent in a private lobby on generals.io.
You need to register at https://generals.io and get your user ID.
"""

import argparse

from generals.remote import autopilot
from generals.agents import ExpanderAgent


def main():
    parser = argparse.ArgumentParser(description="Run agent on generals.io")
    parser.add_argument("--user_id", type=str, required=True, help="Your generals.io user ID")
    parser.add_argument("--lobby_id", type=str, default="bot_test", help="Lobby ID to join")
    args = parser.parse_args()

    agent = ExpanderAgent()
    print(f"Starting {agent.id} agent in lobby '{args.lobby_id}'...")
    autopilot(agent, args.user_id, args.lobby_id)


if __name__ == "__main__":
    main()
