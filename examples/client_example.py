from generals.agents import ExpanderAgent
from generals.remote import GeneralsIOClient


if __name__ == "__main__":
    agent = ExpanderAgent()
    with GeneralsIOClient(agent, "user_id9l") as client:
        # register call will fail when given username is already registered
        # client.register_agent("[Bot]MyEpicUsername")
        client.join_private_lobby("6ngz")
        client.join_game()
