from generals.remote import autopilot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=str, default=...) # Register yourself at generalsio and use this id
parser.add_argument("--lobby_id", type=str, default="psyo") # After you create a private lobby, copy last part of the url
parser.add_argument("--agent_id", type=str, default="Expander") # agent_id should be "registered" in AgentFactory

if __name__ == "__main__":
    args = parser.parse_args()
    autopilot(args.agent_id, args.user_id, args.lobby_id)
