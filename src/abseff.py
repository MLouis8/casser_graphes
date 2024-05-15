with open("data/robust/notweighted/lanes_graph_bc_50p.json", "r") as rfile:
        bcdata = json.load(rfile)

    with open("data/robust/notweighted/lanes_graph_freq_50p.json", "r") as rfile:
        freqdata = json.load(rfile)

    with open("data/robust/notweighted/lanes_graph_deg_50p.json", "r") as rfile:
        degdata = json.load(rfile)

    with open("data/robust/other/efficiency/lanesgraphbc_efficiency_50.json", "r") as rfile:
        bcefficiency = json.load(rfile)

    with open("data/robust/other/efficiency/lanesgraphfreq_efficiency_50.json", "r") as rfile:
        freqefficiency = json.load(rfile)

    with open("data/robust/other/efficiency/lanesgraphdeg_efficiency_50.json", "r") as rfile:
        degefficiency = json.load(rfile)

    t1, t2, t3 = np.arange(len(bcdata)), np.arange(len(freqdata)-1), np.arange(len(degdata))
    y1 = [np.mean(list(attack[1].values())) for attack in bcdata]
    y2 = [np.mean(list(attack[1].values())) for attack in freqdata]
    y3 = [np.mean(list(attack[1].values())) for attack in degdata]
    x1, x2, x3 = np.arange(len(bcefficiency)), np.arange(len(freqefficiency)), np.arange(len(degefficiency))
    w1 = [np.mean(list(efficiency.values())) for efficiency in bcefficiency]
    w2 = [np.mean(list(efficiency.values())) for efficiency in freqefficiency]
    w3 = [np.mean(list(efficiency.values())) for efficiency in degefficiency]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('number of removed edges')
    ax1.set_ylabel('avg eBC', color='tab:red')
    ax1.plot(t1, y1, label="eBC")
    ax1.plot(t2, y2[:len(freqdata)-1], label="freq")
    ax1.plot(t3, y3, label="deg")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('efficiency', color=color)  # we already handled the x-label with ax1
    ax2.plot(x1, w1, label="eBC", color=color)
    ax2.plot(x2, w2, label="freq", color=color)
    ax2.plot(x3, w3, label="deg", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("data/basestrats_notweighted_avgeBC_efficiency.pdf")