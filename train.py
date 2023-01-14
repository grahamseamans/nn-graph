import gym
from net import Net

net = Net(num_inputs=4, num_outputs=2, num_nodes=20, num_edges=100)

env = gym.make("Blackjack-v1")

obs = env.reset()
reward = 0
for _ in range(100):
    action = net.step(obs, reward)
    obs, reward, done, info = env.step(action)
    print(f"action: {action}")
    # print(f"reward: {reward}")

    # print(len(net.G.edges))
    # for edge in net.G.edges():
    #     print(net.G.get_edge_data(*edge))
