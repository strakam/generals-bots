from generals.remote import autopilot
from generals.agents import ExpanderAgent

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=str, default=...) # Register yourself at generals.io and use this id
parser.add_argument("--lobby_id", type=str, default="psyo") # After you create a private lobby, copy last part of the url

if __name__ == "__main__":
    args = parser.parse_args()
    agent = ExpanderAgent()
    autopilot(agent, args.user_id, args.lobby_id)
