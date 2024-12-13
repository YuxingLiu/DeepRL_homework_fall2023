import matplotlib.pyplot as plt
import numpy as np
import os

def CartPole():
    files = ["q2_pg_cartpole_CartPole-v0_13-12-2024_11-35-47",
             "q2_pg_cartpole_rtg_CartPole-v0_13-12-2024_13-13-53",
             "q2_pg_cartpole_na_CartPole-v0_13-12-2024_13-36-35",
             "q2_pg_cartpole_rtg_na_CartPole-v0_13-12-2024_13-39-55",
             "q2_pg_cartpole_lb_CartPole-v0_13-12-2024_13-44-09",
             "q2_pg_cartpole_lb_rtg_CartPole-v0_13-12-2024_13-57-24",
             "q2_pg_cartpole_lb_na_CartPole-v0_13-12-2024_14-01-48",
             "q2_pg_cartpole_lb_rtg_na_CartPole-v0_13-12-2024_14-06-15"]
    data = []
    for file in files:
        data.append(np.genfromtxt(os.path.join("data", file, "data.csv"), names = True, delimiter=' '))

    colors = ['r', 'g', 'b', 'k']
    lines = ['-', '--', ':', '-.']
    labels = ["vanilla", "reward-to-go", "normalize-A", "normalize-A, reward-to-go"]

    plt.figure(figsize=(15, 6))
    for i in range(4):
        plt.subplot(1, 2, 1)
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

        plt.subplot(1, 2, 2)
        plt.plot(data[i + 4]["Train_EnvstepsSoFar"], data[i + 4]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

    plt.subplot(1, 2, 1)
    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "lower right")
    plt.title("Batch size 1000", fontsize = 20)

    plt.subplot(1, 2, 2)
    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "lower right")
    plt.title("Batch size 4000", fontsize = 20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Experiment 1 (CartPole)", fontsize=20)
    plt.savefig('data/1_CartPole.png')


def HalfCheetah():
    files = ["q2_pg_cheetah_HalfCheetah-v4_13-12-2024_17-39-53",
             "q2_pg_cheetah_baseline_HalfCheetah-v4_13-12-2024_17-48-07",
             "q2_pg_cheetah_baseline_HalfCheetah-v4_13-12-2024_17-55-56",
             "q2_pg_cheetah_baseline_HalfCheetah-v4_13-12-2024_18-18-40"]
    data = []
    for file in files:
        data.append(np.genfromtxt(os.path.join("data", file, "data.csv"), names = True, delimiter=' '))

    colors = ['r', 'g', 'b', 'k']
    lines = ['-', '--', ':', '-.']
    labels = ["reward-to-go", "baseline", "-bgs=1", "normalize-A"]

    plt.figure(figsize=(8, 6))
    for i in range(4):
        plt.plot(data[i]["Train_EnvstepsSoFar"], data[i]["Eval_AverageReturn"],
                 color = colors[i], linestyle = lines[i], label = labels[i])

    plt.ylabel("eval average return", fontsize = 16)
    plt.xlabel("env step num", fontsize = 16)
    plt.legend(loc = "lower right")
    plt.title("Batch size 5000", fontsize = 20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Experiment 2 (HalfCheetah)", fontsize=20)
    plt.savefig('data/2_HalfCheetah.png')

def main():
    # CartPole()
    HalfCheetah()

    plt.show()


if __name__ == "__main__":
    main()
