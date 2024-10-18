from generals.agents.random_agent import RandomAgent
from generals.envs.generalsio_client import GeneralsIOClient


if __name__ == '__main__':
    agent = RandomAgent()
    with GeneralsIOClient(agent, 'user_id9l') as client:
        # register call will fail when given username is already registered
        client.register_agent('[Bot]MyEpicUsername')
        client.join_private_game('queueID')
        client.wait_for_game()
