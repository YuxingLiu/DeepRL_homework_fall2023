import matplotlib.pyplot as plt
import numpy as np
import os

def CartPole():
    files = ["hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_23-12-2024_17-10-27",
             "hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_23-12-2024_18-28-47"]
    data = []
    for file in files:
        data.append(np.genfromtxt(os.path.join("data", file, "data.csv"), names = True, delimiter=' '))

    colors = ['r', 'g', 'b', 'k']
    lines = ['-', '--', ':', '-.']
    labels = ["lr: 1e-3", "lr: 0.05"]

    plt.figure(figsize=(18, 6))
    for i in range(2):
        plt.subplot(1, 3, 1)
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

        plt.subplot(1, 3, 2)
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Train_QValues"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

        plt.subplot(1, 3, 3)
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Train_Loss"],
                 color = colors[i], linestyle = lines[i], label = labels[i])


    plt.subplot(1, 3, 1)
    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "best")

    plt.subplot(1, 3, 2)
    plt.ylabel("train Q value", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "best")

    plt.subplot(1, 3, 3)
    plt.ylabel("train loss", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"2.4 CartPole", fontsize=20)
    plt.savefig('data/1_CartPole.png')


def LunarLander():
    files = ["hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-12-2024_17-22-46",
             "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-12-2024_17-44-09",
             "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-12-2024_18-07-49",
             "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-12-2024_18-47-49",
             "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-12-2024_19-24-52",
             "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-12-2024_19-43-43"]
    data = []
    for file in files:
        data.append(np.genfromtxt(os.path.join("data", file, "data.csv"), names = True, delimiter=' '))

    colors = ['r', 'g', 'b', 'k', 'c']
    lines = ['-', '--', ':', '-.', '-']
    labels = ["seed: 1", "seed: 2", "seed: 3"]

    plt.figure(figsize=(15, 6))
    for i in range(3):
        plt.subplot(1, 2, 1)
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

        plt.subplot(1, 2, 2)
        plt.plot(data[i + 3]["Train_EnvstepsSoFar"], data[i + 3]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

    plt.subplot(1, 2, 1)
    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "lower right")
    plt.title("Basic DQN", fontsize = 20)

    plt.subplot(1, 2, 2)
    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "lower right")
    plt.title("Double DQN", fontsize = 20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"2.5 LunarLander", fontsize=20)
    plt.savefig('data/2_LunarLander.png')


def MsPacman():
    files = ["hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_23-12-2024_22-19-10"]
    data = []
    for file in files:
        data.append(np.genfromtxt(os.path.join("data", file, "data.csv"), names = True, delimiter=' '))

    colors = ['r', 'g', 'b', 'k']
    lines = ['-', '--', ':', '-.']
    labels = ["base"]

    plt.figure(figsize=(8, 6))
    for i in range(1):
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"2.5 MsPacman", fontsize=20)
    plt.savefig('data/3_MsPacman.png')


def main():
    # CartPole()
    # LunarLander()
    MsPacman()

    plt.show()


if __name__ == "__main__":
    main()
