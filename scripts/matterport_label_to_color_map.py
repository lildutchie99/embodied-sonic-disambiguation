import random

def hex2rgb(hex):
    hex = hex.lstrip('#')
    return [int(hex[i:i+2], 16) for i in (0, 2, 4)]

r = lambda: random.randint(0,255)
def get_random_hex_color():
    hex_color = "#%02X%02X%02X" % (r(),r(),r())
    return hex_color


mpcat40index2hex = {
    0: "#ffffff",    # void     no
    1: "#aec7e8",    # wall     no
    2: "#708090",    # floor    no
    3: "#98df8a",    # chair    yes
    4: "#c5b0d5",    # door     no
    5: "#ff7f0e",    # table    yes (dining table)
    6: "#d62728",    # picture  no
    7: "#1f77b4",    # cabinet  no
    8: "#bcbd22",    # cushion  no
    9: "#ff9896",    # window   no
    10: "#2ca02c",   # sofa     yes (couch)
    11: "#e377c2",   # bed      yes
    12: "#de9ed6",   # curtain  no
    13: "#9467bd",   # chest_of_drawers no
    14: "#8ca252",   # plant    yes (potted plant)
    15: "#843c39",   # sink     yes
    16: "#9edae5",   # stairs   no
    17: "#9c9ede",   # ceiling  no
    18: "#e7969c",   # toilet   yes
    19: "#637939",   # stool    no
    20: "#8c564b",   # towel    no
    21: "#dbdb8d",   # mirror   no
    22: "#d6616b",   # tv_monitor   yes (tv)
    23: "#cedb9c",   # shower   no
    24: "#e7ba52",   # column   no
    25: "#393b79",   # bathtub  no
    26: "#a55194",   # counter  no
    27: "#ad494a",   # fireplace    no
    28: "#b5cf6b",   # lighting     no
    29: "#5254a3",   # beam         no
    30: "#bd9e39",   # railing      no
    31: "#c49c94",   # shelving     no
    32: "#f7b6d2",   # blinds       no
    33: "#6b6ecf",   # gym_equipment    no
    34: "#ffbb78",   # seating          no
    35: "#c7c7c7",   # board_panel      no
    36: "#8c6d31",   # furniture    no
    37: "#e7cb94",   # appliances   yes (microwave, oven, toaster, refrigerator, hair drier)
    38: "#ce6dbd",   # clothes      no
    39: "#17becf",   # objects      no
    40: "#7f7f7f",   # misc         no
    41: "#000000",   # unlabeled
}
