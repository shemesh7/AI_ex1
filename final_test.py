"""
final_test.py — large-scale test suite for the elevators A* solver.

Groups
  OK        — every correct implementation should return the optimal plan
  PARTIAL   — some implementations may fail (timeout / non-optimal / pruning)
  EXTRA     — existing regression suite from ex1_more_tests (known optima)
  GENERATED — 148 procedurally generated problems with pre-computed optima

Total: 200 test cases.

Usage
    python final_test.py                       # all groups, 60 s timeout each
    python final_test.py --timeout 30          # custom per-test timeout
    python final_test.py --group ok            # only OK group
    python final_test.py --group partial       # only PARTIAL group
    python final_test.py --group extra         # only EXTRA group
    python final_test.py --group generated     # only GENERATED group
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  OK GROUP — every student returns the optimal plan
# ─────────────────────────────────────────────────────────────────────────────

# test_p1 — 4 persons, 2 elevators, bypass available. Optimal: 21
init_state_test_p1 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 6),
        13: (6, 3, 1),
    },
}

# test_p3 — 5 persons, 2 elevators, weight forces single occupancy. Optimal: 16
init_state_test_p3 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6), 8),
        1: (6, (0, 1, 2, 3, 4, 5, 6), 8),
    },
    "Persons": {
        10: (0, 5, 6),
        11: (0, 5, 4),
        12: (6, 5, 0),
        13: (6, 5, 2),
        14: (3, 5, 6),
    },
}

# test_p4 — 4 persons, 2 elevators with overlap, weight allows only 1 each. Optimal: 25
init_state_test_p4 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (4, 5, 6, 7, 8), 10),
    },
    "Persons": {
        10: (0, 6, 8),
        11: (0, 4, 5),
        12: (8, 6, 0),
        13: (8, 5, 3),
    },
}

# competition_1 — 4 persons, 2 elevators, all need transfers. Optimal: 24
init_state_competition_1 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 12),
        1: (6, (3, 4, 5, 6), 12),
    },
    "Persons": {
        10: (0, 4, 5),
        11: (5, 4, 0),
        12: (2, 4, 6),
        13: (5, 4, 1),
    },
}

# competition_2 — shuffle: 3 persons across 3 elevators with overlap. Optimal: 16
init_state_competition_2 = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 300),
        1: (5, (2, 3, 4, 5, 6, 7, 8), 300),
        2: (10, (5, 6, 7, 8, 9, 10), 300),
    },
    "Persons": {
        40: (0, 70, 10),
        41: (2, 70, 8),
        42: (10, 70, 0),
    },
}

TESTPROBS_OK: List[Tuple[str, dict, int]] = [
    ("test_p1",       init_state_test_p1,       21),
    ("test_p3",       init_state_test_p3,        16),
    ("test_p4",       init_state_test_p4,        25),
    ("competition_1", init_state_competition_1,  24),
    ("competition_2", init_state_competition_2,  16),
]

# ─────────────────────────────────────────────────────────────────────────────
#  PARTIAL GROUP — some implementations fail (non-optimal / timeout / pruning)
# ─────────────────────────────────────────────────────────────────────────────

# test_p2 — 4 persons, 3 elevators, extra bypass. Optimal: 22
# Partial: [Timeouts]
init_state_test_p2 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
        2: (4, (7, 2), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 7),
        13: (6, 3, 1),
    },
}

# test_p5 — 4 persons, 3-elevator chain, persons 10/11 need TWO transfers each. Optimal: 31
# Partial: [Pruning, Timeouts]
init_state_test_p5 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (3, 4, 5), 10),
        2: (5, (5, 6, 7, 8), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (4, 3, 7),
        13: (2, 3, 6),
    },
}

# test_p6 — 5 persons, 3 elevators (M3 topology + extra person at transfer floor). Optimal: 24
# Partial: [Non-optimal, Timeouts]
init_state_test_p6 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
        2: (4, (7, 2), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 7),
        13: (6, 3, 1),
        14: (4, 3, 2),
    },
}

# test_p7 — 7-elevator single-floor-reach chain, 3 persons. Optimal: 37
# Partial: [Pruning]
init_state_test_p7 = {
    "height": 14,
    "Elevators": {
        0: (0,  (0, 2),  100),
        1: (2,  (4,),    100),
        2: (4,  (6,),    100),
        3: (6,  (8,),    100),
        4: (8,  (10,),   100),
        5: (10, (12,),   100),
        6: (12, (14,),   100),
    },
    "Persons": {
        10: (0, 8, 14),
        11: (0, 5, 10),
        12: (0, 5, 6),
    },
}

# test_p8 — 4 persons, 3 elevators with a short transfer chain. Optimal: 29
# Partial: [Pruning]
init_state_test_p8 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 9),
        1: (4, (3, 4, 5, 6), 9),
        2: (8, (6, 7, 8),    9),
    },
    "Persons": {
        33: (0, 3, 8),
        34: (8, 3, 0),
        35: (2, 2, 6),
        36: (6, 2, 1),
    },
}

# test_p9 — branching W-chain, single-floor-reach + branch choice at floor 9. Optimal: 24
# Partial: [Pruning]
init_state_test_p9 = {
    "height": 14,
    "Elevators": {
        0: (0,  (0, 3),  100),
        1: (3,  (6,),    100),
        2: (6,  (9,),    100),
        3: (9,  (12,),   100),
        4: (9,  (4,),    100),
        5: (4,  (1,),    100),
    },
    "Persons": {
        20: (0, 10, 12),
        21: (0, 10, 1),
    },
}

# test_p10 — mixed W-chain with normal multi-floor elevator mid-chain. Optimal: 17
# Partial: [Non-optimal, Pruning]
init_state_test_p10 = {
    "height": 16,
    "Elevators": {
        0: (0,  (0, 4),       100),
        1: (4,  (8,),         100),
        2: (8,  (8, 11, 14),  100),
        3: (14, (16,),        100),
    },
    "Persons": {
        30: (0,  10, 16),
        31: (8,  10, 11),
        32: (8,  10, 14),
    },
}

# competition_3 — apartment: 5 persons, express + local + mid elevator. Optimal: 17
# Partial: [Timeouts]
init_state_competition_3 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200),
        1: (8, (0, 2, 4, 6, 8), 300),
        2: (4, (0, 1, 2, 3, 4), 150),
    },
    "Persons": {
        1: (1, 70,  7),
        2: (7, 90,  0),
        3: (0, 110, 8),
        4: (4, 80,  2),
        5: (2, 65,  6),
    },
}

# competition_4 — M1-style: 4 persons, 2 elevators with overlap at floor 4. Optimal: 25
# Partial: [Timeouts]
init_state_competition_4 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 12),
        1: (7, (4, 5, 6, 7),    12),
    },
    "Persons": {
        10: (0, 4, 6),
        11: (6, 4, 1),
        12: (2, 4, 7),
        13: (5, 4, 0),
    },
}

# competition_5 — M1-style variant: 2 elevators with overlap at floors 3,4. Optimal: 24
# Partial: [Non-optimal]
init_state_competition_5 = {
    "height": 8,
    "Elevators": {
        0: (1, (0, 1, 2, 3, 4),       10),
        1: (5, (3, 4, 5, 6, 7, 8),    10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (1, 3, 6),
        13: (6, 3, 1),
    },
}

# competition_6 — H_chain: 4 persons, 3-elevator chain, double-transfer for 40/41. Optimal: 31
# Partial: [Pruning, Timeouts]
init_state_competition_6 = {
    "height": 10,
    "Elevators": {
        0: (0,  (0, 1, 2, 3, 4),    12),
        1: (4,  (4, 5, 6, 7),       12),
        2: (10, (6, 7, 8, 9, 10),   12),
    },
    "Persons": {
        40: (0,  3, 10),
        41: (10, 3, 0),
        42: (2,  3, 8),
        43: (7,  3, 1),
    },
}

# competition_7 — variant of competition_5 with different starting positions. Optimal: 24
# Partial: [Pruning, Non-optimal]
init_state_competition_7 = {
    "height": 9,
    "Elevators": {
        0: (2, (0, 1, 2, 3, 4, 5),    10),
        1: (6, (4, 5, 6, 7, 8, 9),    10),
    },
    "Persons": {
        10: (0, 3, 7),
        11: (7, 3, 0),
        12: (2, 3, 9),
        13: (9, 3, 2),
    },
}

# competition_8 — variant of competition_6: 4 persons, 3-elevator chain. Optimal: 29
# Partial: [Pruning, Timeouts]
init_state_competition_8 = {
    "height": 11,
    "Elevators": {
        0: (0,  (0, 1, 2, 3, 4, 5),   12),
        1: (5,  (4, 5, 6, 7, 8),      12),
        2: (11, (7, 8, 9, 10, 11),    12),
    },
    "Persons": {
        40: (0,  3, 11),
        41: (11, 3, 0),
        42: (3,  3, 6),
        43: (8,  3, 1),
    },
}

# competition_9 — 6 persons, 2 elevators with single overlap floor. Optimal: 33
# Partial: [Timeouts]
init_state_competition_9 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 8),
        1: (4, (4, 5, 6, 7),    8),
    },
    "Persons": {
        10: (0, 4, 7),
        11: (7, 4, 0),
        12: (1, 3, 5),
        13: (5, 3, 1),
        14: (4, 3, 6),
        15: (6, 3, 3),
    },
}

# competition_10 — 4 persons, 3-elevator chain, double-transfer required. Optimal: 28
# Partial: [Pruning, Timeouts]
init_state_competition_10 = {
    "height": 9,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4),    14),
        1: (4, (3, 4, 5, 6, 7),    14),
        2: (9, (7, 8, 9),          14),
    },
    "Persons": {
        40: (0, 4, 9),
        41: (9, 4, 0),
        42: (2, 2, 7),
        43: (7, 2, 1),
    },
}

TESTPROBS_PARTIAL: List[Tuple[str, dict, int]] = [
    ("test_p2",        init_state_test_p2,        22),
    ("test_p5",        init_state_test_p5,        31),
    ("test_p6",        init_state_test_p6,        24),
    ("test_p7",        init_state_test_p7,        37),
    ("test_p8",        init_state_test_p8,        29),
    ("test_p9",        init_state_test_p9,        24),
    ("test_p10",       init_state_test_p10,       17),
    ("competition_3",  init_state_competition_3,  17),
    ("competition_4",  init_state_competition_4,  25),
    ("competition_5",  init_state_competition_5,  24),
    ("competition_6",  init_state_competition_6,  31),
    ("competition_7",  init_state_competition_7,  24),
    ("competition_8",  init_state_competition_8,  29),
    ("competition_9",  init_state_competition_9,  33),
    ("competition_10", init_state_competition_10, 28),
]

# ─────────────────────────────────────────────────────────────────────────────
#  EXTRA GROUP — regression suite from ex1_more_tests
# ─────────────────────────────────────────────────────────────────────────────

_extra: List[Tuple[str, dict, int]] = [
    ("p1",           {"height": 6,  "Elevators": {0: (0,(0,1,2,3),8),     1: (4,(2,4,5,6),10)},   "Persons": {10:(0,3,3),11:(2,4,6),12:(4,5,0)}},               13),
    ("e1",           {"height": 5,  "Elevators": {0: (0,(0,1,2,3,4,5),15),1: (5,(0,1,2,3,4,5),15)},"Persons":{10:(0,3,5),11:(5,3,0),12:(3,3,1)}},               10),
    ("e2",           {"height": 6,  "Elevators": {0: (0,(0,1,2,3),10),    1: (6,(3,4,5,6),10)},    "Persons": {10:(1,3,3),11:(5,3,4),12:(0,3,2)}},               11),
    ("e3",           {"height": 6,  "Elevators": {0: (0,(0,1,2,3,4),10),  1: (6,(2,4,5,6),10)},    "Persons": {10:(0,3,5),11:(6,3,1),12:(3,4,6)}},               18),
    ("e4",           {"height": 5,  "Elevators": {0: (0,(0,1,2,3,4,5),7), 1: (5,(0,1,2,3,4,5),7)}, "Persons":{10:(0,5,5),11:(0,5,3),12:(5,5,0)}},                9),
    ("e5",           {"height": 7,  "Elevators": {0: (0,(0,1,2,3,4),12),  1: (7,(3,4,5,6,7),12)},  "Persons": {10:(1,4,3),11:(6,4,7),12:(0,4,7)}},              13),
    ("m1",           {"height": 6,  "Elevators": {0: (0,(0,1,2,3),12),    1: (6,(3,4,5,6),12)},    "Persons": {10:(0,4,5),11:(5,4,0),12:(2,4,6),13:(5,4,1)}},   24),
    ("m2",           {"height": 8,  "Elevators": {0: (0,(0,1,2,3,4),10),  1: (4,(2,4,6,8),10)},    "Persons": {10:(0,3,8),11:(8,3,0),12:(2,3,6),13:(6,3,1)}},   21),
    ("m3",           {"height": 8,  "Elevators": {0: (0,(0,1,2,3,4),10),  1: (4,(2,4,6,8),10), 2:(4,(7,2),10)}, "Persons":{10:(0,3,8),11:(8,3,0),12:(2,3,7),13:(6,3,1)}}, 22),
    ("m4",           {"height": 6,  "Elevators": {0: (0,(0,1,2,3,4,5,6),8),1:(6,(0,1,2,3,4,5,6),8)},"Persons":{10:(0,5,6),11:(0,5,4),12:(6,5,0),13:(6,5,2),14:(3,5,6)}}, 16),
    ("m5",           {"height": 8,  "Elevators": {0: (0,(0,1,2,3,4),10),  1: (4,(4,5,6,7,8),10)},  "Persons": {10:(0,6,8),11:(0,4,5),12:(8,6,0),13:(8,5,3)}},  25),
    ("c1",           {"height": 4,  "Elevators": {0: (0,(0,1,2,3,4),10),  1: (4,(0,1,2,3,4),10)},  "Persons": {10:(1,2,3),11:(3,2,0)}},                          7),
    ("c2",           {"height": 6,  "Elevators": {0: (0,(0,1,2,3),9),     1: (6,(2,4,5,6),9)},     "Persons": {10:(0,3,6),11:(6,3,1),12:(2,2,5)}},              15),
    ("c3",           {"height": 7,  "Elevators": {0: (0,(0,1,2,3,4),11),  1: (7,(3,4,5,6,7),11)},  "Persons": {10:(0,4,7),11:(1,3,6),12:(6,3,0),13:(7,2,3)}},  21),
    ("e6",           {"height": 4,  "Elevators": {0: (0,(0,1,2,3,4),10),  1: (4,(0,1,2,3,4),10)},  "Persons": {20:(0,2,4),21:(4,2,0)}},                          6),
    ("e7",           {"height": 5,  "Elevators": {0: (0,(0,1,2,3,4,5),12),1: (5,(0,1,2,3,4,5),12)},"Persons":{22:(1,3,5),23:(5,3,1)}},                           6),
    ("m6",           {"height": 7,  "Elevators": {0: (0,(0,1,2,3),10),    1: (7,(3,4,5,6,7),10)},  "Persons": {30:(0,4,7),31:(7,4,0),32:(2,3,6)}},              18),
    ("m7",           {"height": 8,  "Elevators": {0: (0,(0,1,2,3),9),     1: (4,(3,4,5,6),9), 2:(8,(6,7,8),9)}, "Persons":{33:(0,3,8),34:(8,3,0),35:(2,2,6),36:(6,2,1)}}, 29),
    ("h1",           {"height": 9,  "Elevators": {0: (0,(0,1,2,3,4),14),  1: (4,(3,4,5,6,7),14), 2:(9,(7,8,9),14)}, "Persons":{40:(0,4,9),41:(9,4,0),42:(2,2,7),43:(7,2,1)}}, 28),
    ("bottleneck",   {"height": 5,  "Elevators": {0: (0,(0,1,2),500),     1: (0,(0,1,2,3,4,5),500)}, "Persons":{1:(0,80,5),2:(2,70,0)}},                        7),
    ("weight",       {"height": 4,  "Elevators": {0: (0,(0,1,2,3,4),100)},                          "Persons": {1:(0,60,4),2:(0,60,3)}},                         7),
    ("relay",        {"height": 10, "Elevators": {0: (0,(0,1,2,3,4,5),200),1:(10,(5,6,7,8,9,10),200)}, "Persons":{100:(0,75,10)}},                              7),
    ("divided",      {"height": 6,  "Elevators": {0: (0,(0,1,2,3),200),   1: (6,(3,4,5,6),200)},   "Persons": {1:(0,70,6)}},                                    7),
    ("handoff",      {"height": 4,  "Elevators": {0: (0,(0,1,2),1000),    1: (4,(2,3,4),1000), 2:(2,(0,1,2,3,4),100)}, "Persons":{1:(0,500,4)}},                7),
    ("crowded",      {"height": 10, "Elevators": {0: (0,(0,1,2,3,4,5,6,7,8,9,10),250)},            "Persons": {1:(0,80,10),2:(0,80,2),3:(0,80,5)}},              9),
    ("express",      {"height": 8,  "Elevators": {0: (0,(0,1,2,3,4,5,6,7,8),200),1:(0,(0,4,8),200)}, "Persons":{1:(0,70,8),2:(0,70,3)}},                        6),
    ("weight_relay", {"height": 8,  "Elevators": {0: (0,(0,1,2,3,4),1000),1: (8,(4,5,6,7,8),500)}, "Persons": {30:(0,450,8),31:(0,100,8)}},                    13),
    ("shuffle",      {"height": 10, "Elevators": {0: (0,(0,1,2,3,4,5),300),1:(5,(2,3,4,5,6,7,8),300),2:(10,(5,6,7,8,9,10),300)}, "Persons":{40:(0,70,10),41:(2,70,8),42:(10,70,0)}}, 16),
    ("apartment",    {"height": 8,  "Elevators": {0: (0,(0,1,2,3,4,5,6,7,8),200),1:(8,(0,2,4,6,8),300),2:(4,(0,1,2,3,4),150)}, "Persons":{1:(1,70,7),2:(7,90,0),3:(0,110,8),4:(4,80,2),5:(2,65,6)}}, 17),
    # no-solution cases
    ("no_sol_reach",       {"height":8,"Elevators":{0:(0,(0,1,2,3),200),1:(8,(5,6,7,8),200)},"Persons":{10:(0,70,8)}},   0),
    ("no_sol_weight",      {"height":8,"Elevators":{0:(0,(0,1,2),200),1:(4,(3,4,5),50)},     "Persons":{10:(4,100,5)}},  0),
    ("no_sol_goal_access", {"height":8,"Elevators":{0:(0,(0,1,2,3,4),200),1:(4,(4,5,6,7),200)},"Persons":{10:(0,70,8)}}, 0),
]

TESTPROBS_EXTRA: List[Tuple[str, dict, int]] = _extra

# ─────────────────────────────────────────────────────────────────────────────
#  GENERATED GROUP — 148 procedurally generated problems with known optima
# ─────────────────────────────────────────────────────────────────────────────

_generated: List[Tuple[str, dict, int]] = [
    # ── overlap: two elevators sharing a transfer zone ──────────────────────
    ('overlap', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 12), 1: (8, (4, 5, 6, 7, 8), 12)}, 'Persons': {10: (0, 4, 8), 11: (8, 4, 0)}}, 12),
    ('overlap', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 12), 1: (8, (4, 5, 6, 7, 8), 12)}, 'Persons': {10: (0, 4, 8), 11: (8, 4, 0), 12: (2, 3, 7), 13: (7, 3, 1)}}, 24),
    ('overlap', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 12), 1: (10, (5, 6, 7, 8, 9, 10), 12)}, 'Persons': {10: (0, 4, 10), 11: (10, 4, 0), 12: (3, 4, 8)}}, 18),
    ('overlap', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 12), 1: (10, (5, 6, 7, 8, 9, 10), 12)}, 'Persons': {10: (0, 4, 10), 11: (10, 4, 0), 12: (3, 4, 8), 13: (8, 4, 2)}}, 24),
    ('overlap', {'height': 9, 'Elevators': {0: (1, (0, 1, 2, 3, 4, 5), 10), 1: (5, (3, 4, 5, 6, 7, 8, 9), 10)}, 'Persons': {10: (0, 3, 9), 11: (9, 3, 0), 12: (2, 3, 7), 13: (7, 3, 1)}}, 24),
    ('overlap', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 10), 1: (4, (2, 3, 4, 5, 6, 7, 8), 10)}, 'Persons': {10: (0, 3, 8), 11: (8, 3, 0), 12: (2, 3, 7)}}, 16),
    # ── parallel: two identical full-range elevators ─────────────────────────
    ('parallel', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 5), 1: (0, (0, 1, 2, 3, 4, 5, 6), 5)}, 'Persons': {1: (0, 4, 6), 2: (0, 4, 3), 3: (6, 4, 0)}}, 9),
    ('parallel', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 6), 1: (6, (0, 1, 2, 3, 4, 5, 6), 6)}, 'Persons': {1: (0, 5, 6), 2: (0, 5, 4), 3: (6, 5, 0), 4: (6, 5, 2)}}, 12),
    ('parallel', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 10), 1: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 10)}, 'Persons': {1: (0, 6, 8), 2: (0, 6, 5), 3: (8, 6, 0), 4: (8, 6, 3)}}, 13),
    ('parallel', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 8), 1: (5, (0, 1, 2, 3, 4, 5), 8)}, 'Persons': {1: (0, 6, 5), 2: (0, 6, 3), 3: (5, 6, 0)}}, 9),
    # ── asym: asymmetric capacity / reach ────────────────────────────────────
    ('asym', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200), 1: (4, (0, 1, 2, 3, 4), 100)}, 'Persons': {1: (0, 150, 8), 2: (4, 80, 0), 3: (2, 80, 7)}}, 10),
    ('asym', {'height': 7, 'Elevators': {0: (0, (0, 1, 2, 3), 200), 1: (7, (4, 5, 6, 7), 200), 2: (3, (0, 1, 2, 3, 4, 5, 6, 7), 50)}, 'Persons': {1: (0, 30, 7), 2: (7, 30, 0), 3: (0, 180, 4)}}, 0),
    ('asym', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 300), 1: (0, (0, 2, 4, 6), 300)}, 'Persons': {1: (0, 200, 6), 2: (0, 200, 3), 3: (6, 200, 0)}}, 9),
    ('asym', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 50), 1: (5, (5, 6, 7, 8, 9, 10), 50), 2: (0, (0, 5, 10), 200)}, 'Persons': {1: (0, 40, 10), 2: (0, 40, 7), 3: (10, 40, 0)}}, 12),
    # ── 5e: five-elevator chain ───────────────────────────────────────────────
    ('5e', {'height': 10, 'Elevators': {0: (0, (0, 1, 2), 10), 1: (2, (2, 3, 4), 10), 2: (4, (4, 5, 6), 10), 3: (6, (6, 7, 8), 10), 4: (8, (8, 9, 10), 10)}, 'Persons': {10: (0, 5, 10), 11: (10, 5, 0), 12: (2, 4, 8)}}, 36),
    ('5e', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (3, (3, 4, 5, 6), 12), 2: (6, (5, 6, 7, 8), 12), 3: (8, (8, 9, 10), 12), 4: (10, (10, 11, 12), 12)}, 'Persons': {10: (0, 4, 12), 11: (12, 4, 0), 12: (3, 3, 10)}}, 36),
    ('5e', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 10), 1: (4, (2, 4, 6), 10), 2: (6, (6, 7, 8), 10), 3: (8, (8, 9, 10), 10), 4: (0, (0, 5, 10), 20)}, 'Persons': {10: (0, 8, 10), 11: (10, 8, 0), 12: (3, 8, 9)}}, 16),
    # ── tiny1: single transfer, 1 person ──────────────────────────────────────
    ('tiny1', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3), 100), 1: (6, (3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 7),
    ('tiny1', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100), 1: (8, (4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8)}}, 7),
    ('tiny1', {'height': 4, 'Elevators': {0: (0, (0, 1, 2), 100), 1: (4, (2, 3, 4), 100)}, 'Persons': {1: (0, 5, 4)}}, 7),
    ('tiny1', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (10, (5, 6, 7, 8, 9, 10), 100)}, 'Persons': {1: (0, 5, 10)}}, 7),
    ('tiny1', {'height': 6, 'Elevators': {0: (3, (0, 1, 2, 3), 100), 1: (6, (3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 8),
    ('tiny1', {'height': 8, 'Elevators': {0: (4, (0, 1, 2, 3, 4), 100), 1: (8, (4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8)}}, 8),
    # ── btn5: elevator starting between stops ────────────────────────────────
    ('btn5', {'height': 8, 'Elevators': {0: (0, (0, 2, 4, 6, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (0, 5, 4)}}, 6),
    ('btn5', {'height': 8, 'Elevators': {0: (0, (0, 2, 4, 6, 8), 100)}, 'Persons': {1: (8, 5, 0), 2: (8, 5, 4)}}, 7),
    ('btn5', {'height': 8, 'Elevators': {0: (1, (0, 2, 4, 6, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0)}}, 7),
    # ── same: all persons at same floor ───────────────────────────────────────
    ('same', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 10)}, 'Persons': {1: (3, 5, 6), 2: (3, 5, 0), 3: (3, 5, 5)}}, 11),
    ('same', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 10)}, 'Persons': {1: (4, 5, 8), 2: (4, 5, 0), 3: (4, 5, 6)}}, 11),
    ('same', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 6)}, 'Persons': {1: (0, 3, 6), 2: (0, 3, 4), 3: (0, 3, 2)}}, 10),
    # ── goal: persons already at goal floor ──────────────────────────────────
    ('goal', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 0), 2: (0, 5, 6)}}, 3),
    ('goal', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (6, 5, 6), 2: (0, 5, 6), 3: (3, 5, 3)}}, 3),
    # ── 6p2e: 6 persons, 2 elevators ─────────────────────────────────────────
    ('6p2e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 8), 1: (6, (0, 1, 2, 3, 4, 5, 6), 8)}, 'Persons': {1: (0, 5, 6), 2: (0, 5, 4), 3: (6, 5, 0), 4: (6, 5, 2), 5: (3, 5, 6), 6: (3, 5, 0)}}, 20),
    ('6p2e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 7), 1: (4, (4, 5, 6, 7, 8), 7)}, 'Persons': {1: (0, 4, 8), 2: (8, 4, 0), 3: (0, 4, 4), 4: (4, 4, 8), 5: (2, 4, 6), 6: (6, 4, 2)}}, 31),
    # ── nosol: no solution cases ──────────────────────────────────────────────
    ('nosol', {'height': 8, 'Elevators': {0: (0, (0, 1, 2), 100), 1: (8, (6, 7, 8), 100)}, 'Persons': {10: (0, 50, 8)}}, 0),
    ('nosol', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 10)}, 'Persons': {1: (0, 20, 6)}}, 0),
    ('nosol', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3), 100), 1: (8, (5, 6, 7, 8), 100)}, 'Persons': {10: (0, 50, 8)}}, 0),
    ('nosol', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 50), 1: (0, (0, 3, 6), 50)}, 'Persons': {1: (0, 80, 6)}}, 0),
    ('nosol', {'height': 4, 'Elevators': {0: (0, (0, 1, 2), 100)}, 'Persons': {1: (0, 50, 4)}}, 0),
    # ── 3p3e: 3 persons, 3-elevator chain ────────────────────────────────────
    ('3p3e', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (3, (3, 4, 5, 6), 12), 2: (9, (6, 7, 8, 9), 12)}, 'Persons': {1: (0, 8, 9), 2: (9, 8, 0), 3: (3, 8, 7)}}, 25),
    ('3p3e', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (3, (3, 4, 5, 6), 12), 2: (9, (6, 7, 8, 9), 12)}, 'Persons': {1: (0, 8, 9), 2: (9, 8, 0), 3: (6, 8, 3)}}, 22),
    ('3p3e', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7, 8), 12), 2: (12, (8, 9, 10, 11, 12), 12)}, 'Persons': {1: (0, 8, 12), 2: (12, 8, 0), 3: (4, 8, 10)}}, 25),
    ('3p3e', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7, 8), 12), 2: (12, (8, 9, 10, 11, 12), 12)}, 'Persons': {1: (0, 8, 12), 2: (12, 8, 0), 3: (3, 8, 10)}}, 29),
    # ── expr: express + local elevator pair ──────────────────────────────────
    ('expr', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200), 1: (0, (0, 4, 8), 200)}, 'Persons': {1: (0, 150, 8), 2: (0, 150, 4)}}, 6),
    ('expr', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200), 1: (0, (0, 4, 8), 200)}, 'Persons': {1: (0, 100, 8), 2: (0, 100, 4), 3: (0, 100, 2)}}, 9),
    ('expr', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 200), 1: (0, (0, 5, 10), 200)}, 'Persons': {1: (0, 150, 10), 2: (0, 150, 5)}}, 6),
    ('expr', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 200), 1: (0, (0, 4, 8, 12), 200)}, 'Persons': {1: (0, 150, 12), 2: (0, 150, 4), 3: (12, 150, 0)}}, 9),
    # ── sym2: symmetric round-trip ────────────────────────────────────────────
    ('sym2', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 10)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0), 3: (4, 5, 8)}}, 9),
    ('sym2', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 10)}, 'Persons': {1: (0, 5, 6), 2: (6, 5, 0)}}, 6),
    ('sym2', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 12)}, 'Persons': {1: (0, 4, 10), 2: (10, 4, 0), 3: (5, 4, 10), 4: (5, 4, 0)}}, 11),
    # ── nr1: non-reach initial floor (chain-style) ────────────────────────────
    ('nr1', {'height': 4, 'Elevators': {0: (2, (4,), 100)}, 'Persons': {1: (4, 5, 0)}}, 0),
    ('nr1', {'height': 4, 'Elevators': {0: (0, (0, 2, 4), 100)}, 'Persons': {1: (0, 5, 4)}}, 3),
    ('nr1', {'height': 6, 'Elevators': {0: (0, (0, 2, 4), 100), 1: (6, (4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 7),
    ('nr1', {'height': 6, 'Elevators': {0: (1, (0, 2, 4), 100), 1: (6, (4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 8),
    ('nr1', {'height': 8, 'Elevators': {0: (0, (0, 2, 4), 100), 1: (4, (4, 6, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0)}}, 12),
    ('nr1', {'height': 8, 'Elevators': {0: (0, (0, 2, 4), 100), 1: (4, (4, 6, 8), 100)}, 'Persons': {1: (0, 5, 8)}}, 6),
    ('nr1', {'height': 10, 'Elevators': {0: (0, (0, 2, 4, 6), 100), 1: (6, (6, 8, 10), 100)}, 'Persons': {1: (0, 5, 10), 2: (10, 5, 0)}}, 12),
    ('nr1', {'height': 8, 'Elevators': {0: (2, (0, 4, 8), 100)}, 'Persons': {1: (0, 5, 8)}}, 4),
    ('nr1', {'height': 6, 'Elevators': {0: (3, (0, 3, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 4),
    ('nr1', {'height': 6, 'Elevators': {0: (1, (0, 2, 4, 6), 100)}, 'Persons': {1: (0, 5, 6), 2: (6, 5, 0)}}, 7),
    # ── wt: weight-constrained loading ───────────────────────────────────────
    ('wt', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 10)}, 'Persons': {1: (0, 6, 6), 2: (0, 6, 4), 3: (6, 6, 0)}}, 9),
    ('wt', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 10)}, 'Persons': {1: (0, 5, 6), 2: (0, 5, 4), 3: (0, 5, 2)}}, 10),
    ('wt', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 10)}, 'Persons': {1: (0, 4, 8), 2: (0, 4, 5), 3: (0, 4, 3)}}, 10),
    ('wt', {'height': 4, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 8)}, 'Persons': {1: (0, 5, 4), 2: (0, 5, 2), 3: (0, 5, 1)}}, 11),
    ('wt', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 20), 1: (0, (0, 1, 2, 3, 4, 5, 6), 20)}, 'Persons': {1: (0, 15, 6), 2: (0, 15, 4), 3: (0, 15, 2)}}, 10),
    ('wt', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 15), 1: (8, (0, 1, 2, 3, 4, 5, 6, 7, 8), 15)}, 'Persons': {1: (0, 10, 8), 2: (0, 10, 5), 3: (8, 10, 0)}}, 9),
    ('wt', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100), 1: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 60, 6), 2: (0, 60, 4), 3: (0, 60, 2)}}, 10),
    # ── triv1: trivial single-elevator, various heights ───────────────────────
    ('triv1', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100)}, 'Persons': {1: (0, 5, 5)}}, 3),
    ('triv1', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 3),
    ('triv1', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8)}}, 3),
    ('triv1', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 100)}, 'Persons': {1: (0, 5, 10)}}, 3),
    ('triv1', {'height': 3, 'Elevators': {0: (0, (0, 1, 2, 3), 100)}, 'Persons': {1: (0, 5, 3)}}, 3),
    ('triv1', {'height': 4, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100)}, 'Persons': {1: (1, 5, 4)}}, 4),
    ('triv1', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (2, 5, 5)}}, 4),
    ('triv1', {'height': 8, 'Elevators': {0: (4, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8)}}, 4),
    ('triv1', {'height': 6, 'Elevators': {0: (3, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6)}}, 4),
    ('triv1', {'height': 4, 'Elevators': {0: (4, (0, 1, 2, 3, 4), 100)}, 'Persons': {1: (0, 5, 4)}}, 4),
    # ── triv2: trivial 2-person, 1-elevator ─────────────────────────────────
    ('triv2', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6), 2: (0, 5, 3)}}, 6),
    ('triv2', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0)}}, 6),
    ('triv2', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6), 2: (6, 5, 0)}}, 6),
    ('triv2', {'height': 4, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100)}, 'Persons': {1: (0, 5, 4), 2: (0, 5, 2)}}, 6),
    ('triv2', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 100)}, 'Persons': {1: (0, 5, 10), 2: (5, 5, 10)}}, 6),
    # ── med3e: medium 3-elevator chain ────────────────────────────────────────
    ('med3e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (4, (3, 4, 5, 6), 12), 2: (8, (6, 7, 8), 12)}, 'Persons': {1: (0, 8, 8), 2: (8, 8, 0)}}, 19),
    ('med3e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (4, (3, 4, 5, 6), 12), 2: (8, (6, 7, 8), 12)}, 'Persons': {1: (0, 8, 8), 2: (8, 8, 0), 3: (3, 8, 7)}}, 26),
    ('med3e', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7), 12), 2: (10, (7, 8, 9, 10), 12)}, 'Persons': {1: (0, 8, 10), 2: (10, 8, 0)}}, 18),
    ('med3e', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7), 12), 2: (10, (7, 8, 9, 10), 12)}, 'Persons': {1: (0, 8, 10), 2: (10, 8, 0), 3: (4, 8, 8)}}, 25),
    ('med3e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2), 12), 1: (2, (2, 3, 4), 12), 2: (6, (4, 5, 6), 12)}, 'Persons': {1: (0, 8, 6), 2: (6, 8, 0)}}, 18),
    ('med3e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2), 12), 1: (2, (2, 3, 4), 12), 2: (6, (4, 5, 6), 12)}, 'Persons': {1: (0, 8, 6), 2: (6, 8, 0), 3: (2, 8, 5)}}, 25),
    # ── 4ec: 4-elevator chain ─────────────────────────────────────────────────
    ('4ec', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (3, (3, 4, 5, 6), 12), 2: (6, (6, 7, 8), 12), 3: (9, (8, 9), 12)}, 'Persons': {10: (0, 4, 9), 11: (9, 4, 0)}}, 24),
    ('4ec', {'height': 16, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 14), 1: (4, (4, 5, 6, 7, 8), 14), 2: (8, (8, 9, 10, 11, 12), 14), 3: (12, (12, 13, 14, 15, 16), 14)}, 'Persons': {40: (0, 4, 16), 41: (16, 4, 0)}}, 24),
    ('4ec', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3), 10), 1: (3, (3, 4, 5, 6), 10), 2: (6, (5, 6, 7, 8, 9), 10), 3: (9, (9, 10, 11, 12), 10)}, 'Persons': {10: (0, 3, 12), 11: (12, 3, 0), 12: (2, 3, 9)}}, 31),
    # ── 1e3p: single elevator, 3 persons in sequence ─────────────────────────
    ('1e3p', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6), 2: (6, 5, 3), 3: (3, 5, 0)}}, 9),
    ('1e3p', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 4), 3: (4, 5, 0)}}, 9),
    ('1e3p', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100)}, 'Persons': {1: (0, 5, 5), 2: (5, 5, 2), 3: (2, 5, 0)}}, 9),
    # ── 1p2e: single person with 2-elevator chain ────────────────────────────
    ('1p2e', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3), 100), 1: (5, (2, 3, 4, 5), 100)}, 'Persons': {1: (0, 5, 5)}}, 7),
    ('1p2e', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3), 100), 1: (5, (2, 3, 4, 5), 100)}, 'Persons': {1: (1, 5, 4)}}, 8),
    ('1p2e', {'height': 7, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100), 1: (7, (3, 4, 5, 6, 7), 100)}, 'Persons': {1: (1, 5, 6)}}, 8),
    ('1p2e', {'height': 7, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100), 1: (7, (3, 4, 5, 6, 7), 100)}, 'Persons': {1: (2, 5, 7)}}, 8),
    ('1p2e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (8, (3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 7)}}, 7),
    ('1p2e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (8, (3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (1, 5, 6)}}, 8),
    ('1p2e', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (9, (4, 5, 6, 7, 8, 9), 100)}, 'Persons': {1: (0, 5, 9)}}, 7),
    ('1p2e', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (9, (4, 5, 6, 7, 8, 9), 100)}, 'Persons': {1: (2, 5, 7)}}, 8),
    # ── bypass2p: express bypass elevator + 2 chain elevators ────────────────
    ('bypass2p', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100), 1: (8, (4, 5, 6, 7, 8), 100), 2: (0, (0, 2, 4, 6, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0)}}, 6),
    ('bypass2p', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3), 100), 1: (6, (3, 4, 5, 6), 100), 2: (0, (0, 3, 6), 100)}, 'Persons': {1: (0, 5, 6), 2: (6, 5, 0)}}, 6),
    ('bypass2p', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (10, (5, 6, 7, 8, 9, 10), 100), 2: (0, (0, 5, 10), 100)}, 'Persons': {1: (0, 5, 10), 2: (10, 5, 0)}}, 6),
    # ── double_xfr: two persons needing two elevator transfers each ───────────
    ('double_xfr', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3), 12), 1: (3, (3, 4, 5, 6), 12), 2: (9, (6, 7, 8, 9), 12)}, 'Persons': {1: (0, 5, 9), 2: (9, 5, 0)}}, 18),
    ('double_xfr', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7, 8), 12), 2: (12, (8, 9, 10, 11, 12), 12)}, 'Persons': {1: (0, 5, 12), 2: (12, 5, 0)}}, 18),
    ('double_xfr', {'height': 8, 'Elevators': {0: (0, (0, 1, 2), 10), 1: (2, (2, 3, 4, 5, 6), 10), 2: (8, (6, 7, 8), 10)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0)}}, 18),
    # ── 1e_multi: single elevator, 3 persons with capacity planning ───────────
    ('1e_multi', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 4, 6), 2: (0, 4, 3), 3: (6, 4, 0)}}, 9),
    ('1e_multi', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 3, 8), 2: (8, 3, 5), 3: (5, 3, 0)}}, 9),
    ('1e_multi', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100)}, 'Persons': {1: (0, 5, 5), 2: (5, 5, 2), 3: (2, 5, 0)}}, 9),
    ('1e_multi', {'height': 4, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100)}, 'Persons': {1: (0, 3, 4), 2: (4, 3, 2), 3: (2, 3, 0)}}, 9),
    # ── 3e2p: 3 elevators, 2 persons (including express bypass) ──────────────
    ('3e2p', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100), 1: (8, (4, 5, 6, 7, 8), 100), 2: (0, (0, 8), 50)}, 'Persons': {1: (0, 4, 8), 2: (8, 4, 0)}}, 6),
    ('3e2p', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3), 100), 1: (6, (3, 4, 5, 6), 100), 2: (0, (0, 6), 50)}, 'Persons': {1: (0, 3, 6), 2: (6, 3, 0)}}, 6),
    ('3e2p', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100), 1: (10, (5, 6, 7, 8, 9, 10), 100), 2: (0, (0, 5, 10), 50)}, 'Persons': {1: (2, 3, 8), 2: (8, 3, 2)}}, 14),
    # ── partial_done: some persons already at their goal ─────────────────────
    ('partial_done', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (2, 5, 2), 2: (0, 5, 6), 3: (6, 5, 0)}}, 6),
    ('partial_done', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7, 8), 12)}, 'Persons': {1: (4, 5, 4), 2: (4, 5, 4), 3: (0, 5, 8), 4: (8, 5, 0)}}, 12),
    ('partial_done', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (3, 5, 3), 2: (3, 5, 3), 3: (0, 5, 6)}}, 3),
    # ── 4p2e: 4 persons, 2 tight-capacity elevators ──────────────────────────
    ('4p2e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 7), 1: (6, (0, 1, 2, 3, 4, 5, 6), 7)}, 'Persons': {1: (0, 5, 6), 2: (0, 5, 4), 3: (6, 5, 0), 4: (6, 5, 2)}}, 12),
    ('4p2e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 6), 1: (4, (4, 5, 6, 7, 8), 6)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0), 3: (0, 5, 4), 4: (4, 5, 8)}}, 18),
    ('4p2e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3), 10), 1: (6, (3, 4, 5, 6), 10)}, 'Persons': {1: (0, 6, 6), 2: (6, 6, 0), 3: (1, 5, 5), 4: (5, 5, 1)}}, 25),
    ('4p2e', {'height': 7, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 10), 1: (7, (4, 5, 6, 7), 10)}, 'Persons': {1: (0, 6, 7), 2: (7, 6, 0), 3: (1, 5, 5), 4: (5, 5, 2)}}, 25),
    # ── chain_multi: 3-elevator chain with persons entering at different links ─
    ('chain_multi', {'height': 9, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 12), 1: (4, (4, 5, 6, 7), 12), 2: (9, (7, 8, 9), 12)}, 'Persons': {1: (0, 8, 9), 2: (4, 8, 9), 3: (7, 8, 0)}}, 23),
    ('chain_multi', {'height': 12, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 14), 1: (4, (3, 4, 5, 6, 7, 8), 14), 2: (12, (8, 9, 10, 11, 12), 14)}, 'Persons': {1: (0, 8, 12), 2: (4, 8, 12), 3: (8, 8, 0)}}, 23),
    ('chain_multi', {'height': 8, 'Elevators': {0: (0, (0, 1, 2), 8), 1: (2, (2, 3, 4, 5, 6), 8), 2: (8, (6, 7, 8), 8)}, 'Persons': {1: (0, 5, 8), 2: (2, 5, 8), 3: (6, 5, 0)}}, 23),
    # ── crowded1e: single elevator at capacity ────────────────────────────────
    ('crowded1e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 50)}, 'Persons': {1: (0, 40, 8), 2: (0, 40, 5), 3: (0, 40, 3)}}, 11),
    ('crowded1e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 20)}, 'Persons': {1: (0, 10, 6), 2: (0, 10, 3), 3: (6, 10, 0), 4: (6, 10, 2)}}, 12),
    # ── up3: 3 persons all going upward, tight capacity ───────────────────────
    ('up3', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 20), 1: (6, (4, 5, 6), 20)}, 'Persons': {1: (0, 7, 6), 2: (0, 7, 3), 3: (0, 7, 5)}}, 16),
    ('up3', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 20), 1: (8, (4, 5, 6, 7, 8), 20)}, 'Persons': {1: (0, 7, 8), 2: (0, 7, 4), 3: (0, 7, 7)}}, 16),
    ('up3', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3), 20), 1: (5, (3, 4, 5), 20)}, 'Persons': {1: (0, 7, 5), 2: (0, 7, 2), 3: (0, 7, 4)}}, 16),
    # ── zigzag1e: single elevator, up-down-up persons ─────────────────────────
    ('zigzag1e', {'height': 7, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7), 100)}, 'Persons': {1: (0, 5, 7), 2: (7, 5, 0), 3: (3, 5, 7), 4: (7, 5, 3)}}, 12),
    ('zigzag1e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (0, 5, 6), 2: (6, 5, 0), 3: (2, 5, 4), 4: (4, 5, 2)}}, 13),
    ('zigzag1e', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100)}, 'Persons': {1: (0, 5, 5), 2: (5, 5, 0), 3: (1, 5, 3), 4: (3, 5, 1)}}, 13),
    ('zigzag1e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (0, 5, 8), 2: (8, 5, 0), 3: (4, 5, 8), 4: (0, 5, 4)}}, 11),
    # ── heavy2e: two full-range elevators, heavy persons ─────────────────────
    ('heavy2e', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 80), 1: (0, (0, 1, 2, 3, 4, 5, 6), 80)}, 'Persons': {1: (0, 40, 6), 2: (0, 40, 6), 3: (0, 40, 3)}}, 8),
    ('heavy2e', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 80), 1: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 80)}, 'Persons': {1: (0, 40, 8), 2: (0, 40, 8), 3: (0, 40, 4)}}, 8),
    ('heavy2e', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 80), 1: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 80)}, 'Persons': {1: (0, 40, 10), 2: (0, 40, 10), 3: (0, 40, 5)}}, 8),
    # ── same_dst: all persons heading to same floor ───────────────────────────
    ('same_dst', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 5)}, 'Persons': {1: (0, 1, 6), 2: (0, 1, 6), 3: (0, 1, 6)}}, 7),
    ('same_dst', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 8)}, 'Persons': {1: (0, 2, 8), 2: (0, 2, 8), 3: (0, 2, 8)}}, 7),
    ('same_dst', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 6)}, 'Persons': {1: (0, 2, 5), 2: (0, 2, 5), 3: (0, 2, 5)}}, 7),
    # ── many_e1p: many elevator choices for a single person ──────────────────
    ('many_e1p', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3), 50), 1: (8, (4, 5, 6, 7, 8), 50), 2: (0, (0, 8), 50), 3: (0, (0, 4, 8), 50)}, 'Persons': {1: (0, 40, 8)}}, 3),
    ('many_e1p', {'height': 10, 'Elevators': {0: (0, (0, 1, 2, 3), 50), 1: (10, (4, 5, 6, 7, 8), 50), 2: (0, (0, 10), 50), 3: (0, (0, 5, 10), 50)}, 'Persons': {1: (0, 40, 10)}}, 3),
    ('many_e1p', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3), 50), 1: (6, (4, 5, 6), 50), 2: (0, (0, 6), 50), 3: (0, (0, 3, 6), 50)}, 'Persons': {1: (0, 40, 6)}}, 3),
    # ── one_done: one person already at goal ──────────────────────────────────
    ('one_done', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 100)}, 'Persons': {1: (6, 5, 6), 2: (0, 5, 6)}}, 3),
    ('one_done', {'height': 8, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 100)}, 'Persons': {1: (8, 5, 8), 2: (0, 5, 8)}}, 3),
    ('one_done', {'height': 5, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5), 100)}, 'Persons': {1: (5, 5, 5), 2: (0, 5, 5)}}, 3),
    # ── filler: simple baseline cases ─────────────────────────────────────────
    ('filler1', {'height': 4, 'Elevators': {0: (0, (0, 1, 2, 3, 4), 100)}, 'Persons': {1: (0, 5, 4)}}, 3),
    ('filler2', {'height': 4, 'Elevators': {0: (0, (0, 1, 2), 50), 1: (4, (2, 3, 4), 50)}, 'Persons': {1: (0, 30, 4), 2: (4, 30, 0)}}, 12),
    ('filler3', {'height': 6, 'Elevators': {0: (0, (0, 1, 2, 3, 4, 5, 6), 30)}, 'Persons': {1: (0, 15, 6), 2: (6, 15, 0), 3: (3, 15, 6)}}, 9),
]

TESTPROBS_GENERATED: List[Tuple[str, dict, int]] = _generated

# ─────────────────────────────────────────────────────────────────────────────
#  Solver worker (top-level so multiprocessing can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_worker(init_state: dict) -> Tuple[Optional[int], Optional[int]]:
    """Return (plan_length, nodes_expanded) or (None, None) for no-solution."""
    import ex1_331050591
    import search
    try:
        prob = ex1_331050591.create_elevators_problem(init_state)
        result = search.astar_search(prob, prob.h_astar)
        if result is None:
            return None, None
        node, expanded = result
        if not isinstance(node, search.Node):
            return None, None
        actions = [n.action for n in node.path()[::-1]][1:]
        return len(actions), expanded
    except Exception as exc:  # noqa: BLE001
        return -1, str(exc)


def solve_with_timeout(
    init_state: dict, timeout_s: float
) -> Tuple[object, Optional[int]]:
    """
    Returns (cost_or_tag, expanded_or_msg).
    cost_or_tag is an int for a plan, None for no-solution, or a string tag
    such as "TIMEOUT" / "ERROR".
    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_solve_worker, init_state)
        try:
            cost, extra = future.result(timeout=timeout_s)
            return cost, extra
        except FutTimeout:
            return "TIMEOUT", None
        except Exception as exc:  # noqa: BLE001
            return "ERROR", str(exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Result formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"   # solved but suboptimal

_TICK = "✓"
_CROSS = "✗"
_BANG = "!"

def _verdict(cost, expected: int) -> Tuple[str, str]:
    """Return (verdict_tag, detail_string)."""
    if isinstance(cost, str):          # TIMEOUT / ERROR
        return FAIL, cost
    if cost is None:                   # no solution found
        if expected == 0:
            return PASS, "NO-SOL (expected)"
        return FAIL, "NO-SOL (unexpected)"
    if expected == 0:
        return FAIL, f"GOT {cost} (expected no solution)"
    if cost == expected:
        return PASS, f"cost={cost}"
    return FAIL, f"cost={cost} (expected {expected})"


# ─────────────────────────────────────────────────────────────────────────────
#  Suite runner
# ─────────────────────────────────────────────────────────────────────────────

Row = Tuple[str, str, str, str, str, float]   # name, verdict, detail, expanded, group, elapsed


def run_suite(
    suite: List[Tuple[str, dict, int]],
    group_label: str,
    timeout_s: float,
    verbose: bool = True,
) -> Tuple[List[Row], int, int]:
    """Run a suite and return (rows, n_pass, n_total)."""
    rows: List[Row] = []
    n_pass = 0

    if verbose:
        print(f"\n{'='*72}")
        print(f"  GROUP: {group_label}  ({len(suite)} tests, timeout={timeout_s}s each)")
        print(f"{'='*72}")

    for name, init_state, expected in suite:
        t0 = time.perf_counter()
        cost, extra = solve_with_timeout(init_state, timeout_s)
        elapsed = time.perf_counter() - t0

        verdict, detail = _verdict(cost, expected)
        expanded_str = str(extra) if isinstance(extra, int) else (extra or "-")
        if verdict == PASS:
            n_pass += 1

        rows.append((name, verdict, detail, expanded_str, group_label, elapsed))

        if verbose:
            symbol = _TICK if verdict == PASS else _CROSS
            print(
                f"  {symbol} {name:<26} {detail:<30} "
                f"exp={expanded_str:<8} {elapsed:6.2f}s"
            )

    return rows, n_pass, len(suite)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Elevator A* test suite")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="per-test timeout in seconds (default: 60)")
    parser.add_argument("--group", choices=["ok", "partial", "extra", "generated", "all"],
                        default="all", help="which group(s) to run")
    args = parser.parse_args()

    all_rows: List[Row] = []
    total_pass = 0
    total_tests = 0

    suites = {
        "ok":        (TESTPROBS_OK,        "OK (all students pass)"),
        "partial":   (TESTPROBS_PARTIAL,   "PARTIAL (some students fail)"),
        "extra":     (TESTPROBS_EXTRA,     "EXTRA (regression)"),
        "generated": (TESTPROBS_GENERATED, "GENERATED (procedural)"),
    }

    active = list(suites.keys()) if args.group == "all" else [args.group]

    wall_start = time.perf_counter()

    for key in active:
        suite, label = suites[key]
        rows, n_pass, n_total = run_suite(suite, label, args.timeout)
        all_rows.extend(rows)
        total_pass  += n_pass
        total_tests += n_total

    wall_elapsed = time.perf_counter() - wall_start

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(
        f"  {'Name':<26} {'Group':<10} {'Verdict':<6} {'Detail':<30} {'Sec':>6}"
    )
    print(f"  {'-'*26} {'-'*10} {'-'*6} {'-'*30} {'-'*6}")
    for name, verdict, detail, _, group, elapsed in all_rows:
        sym = _TICK if verdict == PASS else _CROSS
        print(
            f"  {sym} {name:<26} {group:<10} {verdict:<6} {detail:<30} {elapsed:6.2f}"
        )

    # ── Totals ───────────────────────────────────────────────────────────────
    n_fail = total_tests - total_pass
    print(f"\n  Passed : {total_pass}/{total_tests}")
    print(f"  Failed : {n_fail}/{total_tests}")
    print(f"  Wall   : {wall_elapsed:.1f}s")

    if n_fail:
        print("\n  FAILED TESTS:")
        for name, verdict, detail, _, group, _ in all_rows:
            if verdict != PASS:
                print(f"    [{group}] {name} — {detail}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
