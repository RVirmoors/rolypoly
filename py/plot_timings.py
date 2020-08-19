import matplotlib.pyplot as plt
import numpy as np

timings = np.array([])
data1 = np.array(np.loadtxt(
    "data/takes/s2s-swing-A3-noB/1.csv", delimiter=","))
data2 = np.array(np.loadtxt(
    "data/takes/s2s-swing-A3-noB/2.csv", delimiter=","))
data3 = np.array(np.loadtxt(
    "data/takes/s2s-swing-A3-noB/3.csv", delimiter=","))
data4 = np.array(np.loadtxt(
    "data/takes/s2s-swing-A3-noB/4.csv", delimiter=","))
timings = np.stack((data1, data2, data3, data4))
# 13 pos in bar
# 15 d_g_diff
# 17 y_hat

fig, ax = plt.subplots()

score = timings[0, 5:37, 0] + timings[0, 5:37, 13]
# print(score)
ax.vlines(score, 0, 35, linestyles='dotted')

for i in range(4):
    stems = score + timings[i, 5:37, 17]

    markerline, stemline, baseline = ax.stem(stems, [i * 10 + 7] * stems.shape[0],
                                             bottom=i * 10,
                                             # linefmt="C1-",  # basefmt="k-",
                                             use_line_collection=True)

    barDur = timings[i, 5:37, 15]
    bars = [(stems[j], barDur[j]) for j in range(stems.shape[0])]
    # print(bars)

    ax.broken_barh(bars, (i * 10, 5),
                   facecolors='grey')  # [(0, 1), (2, 3)]


ax.set_ylim(0, 38)
ax.set_xlim(1, 4.8)
ax.set_xlabel('bars')
ax.set_yticks([0, 10, 20, 30])
ax.set_yticklabels(['baseline', 'iter. 1', 'iter. 2', 'iter. 3'])
ax.grid(False)
plt.show()
