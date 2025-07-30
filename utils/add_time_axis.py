import numpy as np
import matplotlib.pyplot as plt

def years_to_mtu(t):
    return 360*t/5

def days_to_mtu(t):
    return t/5

def mtu_to_days(t):
    return t*5

def mtu_to_years(t):
    return t*5/360

def add_axis(ax, mtu_ticks, new_ticks, xlabel='Time (~Atmospheric Days)'):
    ax2 = ax.twiny()

    ax2.set_xticks(mtu_ticks)
    ax2.set_xticklabels([int(new_tick) for new_tick in new_ticks])

    ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 36))
    ax2.set_xlabel(xlabel)
    ax2.set_xlim(ax.get_xlim())

    plt.tight_layout()


def add_axis_weather(ax, max_days = 15., step_days = 3.):
    day_ticks = np.arange(0., max_days, step_days)
    mtu_ticks = days_to_mtu(day_ticks)
    add_axis(ax, mtu_ticks, day_ticks, xlabel='Time (~Atmospheric Days)')

def add_axis_climate(ax, max_years = 15, step_years = 5.):
    year_ticks = np.arange(0., max_years, step_years)
    mtu_ticks = years_to_mtu(year_ticks)
    add_axis(ax, mtu_ticks, year_ticks, xlabel='Time (~Atmospheric Years)')



