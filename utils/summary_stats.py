import numpy as numpy
import torch 

def summary_stats(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "15.85%": v.kthvalue(int(len(v) * 0.1585), dim=0)[0],
            "84.13%": v.kthvalue(int(len(v) * 0.8413), dim=0)[0],
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats