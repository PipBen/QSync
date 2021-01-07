t_cm = t / (2 * constant.pi)

t_ps = (t_cm * 1e12) / (100 * constant.c)

t_w = t_ps / (1e12 / (omega * 100 * constant.c))

dtperps = (100 * constant.c * 2 * constant.pi * 1e-12) / dt

itvl = 5