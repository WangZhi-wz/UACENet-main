import torch
from torch import tensor

if __name__ == '__main__':
    print('#### Test Case ###')
    x = tensor([[0.]])

    x = torch.sigmoid(x)

    # uacanet
    fg = x
    p = fg - .5
    # foreground
    fg = torch.clip(p, 0, 1)
    # background
    bg = torch.clip(-p, 0, 1)
    # edge
    cg = .5 - torch.abs(p)

    # ours
    # foreground
    fg_ours = x
    # background
    bg_ours = -x
    # edge
    dist = torch.abs(x - 0.5)
    cg_ours = 1 - (dist / 0.5)

    print("uaca: ", fg, bg, cg)
    print("ours: ", fg_ours, bg_ours, cg_ours)



