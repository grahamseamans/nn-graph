import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib
import networkx as nx

matplotlib.use("TkAgg")

fig = plt.figure()
net = fig.add_subplot(111)


def animate(i):
    f = open("data.txt")

    N = []
    for line in f:
        N.append(line.strip())

    G = nx.Graph()

    for e in N:
        G.add_node(e)

    plt.clf()
    plt.cla()
    nx.draw(G)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
