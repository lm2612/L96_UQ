def plotcolor(name):
    if "IC" in name:
        return "pink"
    elif "aleatoric" in name:
        return "seagreen"
    elif "epistemic" in name:
        return "darkorchid"
    elif "both" in name:
        return "dimgray"
    elif "deterministic" in name:
        return "midnightblue"
    elif "Truth" in name:
        return "black"
    elif "Dropout" in name:
        return "pink"
    elif "Aleatoric" in name:
        return "darkgoldenrod"
    