from generals.remote import autopilot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=str, default="user_id9l")
parser.add_argument("--lobby_id", type=str, default="elj2")
parser.add_argument("--agent_id", type=str, default="Expander") # agent_id should be "registered" in AgentFactory

if __name__ == "__main__":
    args = parser.parse_args()
    autopilot(args.agent_id, args.user_id, args.lobby_id)
