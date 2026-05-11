"""
Exhaustive admissibility test for h_astar — 10,000+ instances.

Three levels of verification:

  LEVEL 1 – OPTIMALITY (all instances)
      A*(h_astar) cost  ==  UCS cost.

  LEVEL 2 – PATH CHECK (all instances)
      For every node on the A* optimal path:  h(n) <= opt_cost - g(n).

  LEVEL 3 – EXHAUSTIVE h* CHECK (small instances)
      Forward-BFS the ENTIRE reachable state space, backward-BFS
      from goals to compute true h*(s) for EVERY reachable state,
      then verify h(s) <= h*(s).

Also checks consistency h(n) <= c(n,a,n') + h(n') and h(goal)==0.

Phase breakdown (~10,000+ instances total):
  Phase 1:   11  known problems with expected optimal costs
  Phase 2: 3000  purely random problems
  Phase 3: 3000  targeted edge-case problems
  Phase 4: 2000  larger stress-test problems (3-4 persons, 2-3 elevators)
  Phase 5: 2000  adversarial corner-case problems

Usage:
    python test_admissibility_10k.py
"""

import time
import random as rng
import sys
import heapq
from collections import deque, Counter

import ex1
import search
from search import Node


# ═════════════════════════════════════════════════════════════════════════════
#  RANDOM PROBLEM GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def random_contiguous_floors(height, min_count=2):
    lo = rng.randint(0, height - 1)
    hi = rng.randint(lo + 1, height)
    return tuple(range(lo, hi + 1))


def random_sparse_floors(height, min_count=2):
    all_floors = list(range(height + 1))
    k = rng.randint(min_count, len(all_floors))
    return tuple(sorted(rng.sample(all_floors, k)))


def generate_random_problem(seed=None):
    """Generate a purely random elevator problem."""
    if seed is not None:
        rng.seed(seed)

    height = rng.randint(3, 7)
    num_elevators = rng.randint(1, 3)
    num_persons = rng.randint(1, 3)

    elevators = {}
    for eid in range(num_elevators):
        if rng.random() < 0.6:
            floors = random_contiguous_floors(height)
        else:
            floors = random_sparse_floors(height, min_count=2)
        current_floor = rng.choice(floors)
        max_weight = rng.choice([5, 8, 10, 12, 15, 20])
        elevators[eid] = (current_floor, floors, max_weight)

    persons = {}
    for pid in range(10, 10 + num_persons):
        start_floor = rng.randint(0, height)
        goal_floor = rng.randint(0, height)
        weight = rng.randint(1, 6)
        persons[pid] = (start_floor, weight, goal_floor)

    return {
        "height": height,
        "Elevators": elevators,
        "Persons": persons,
    }


def generate_targeted_problem(seed=None):
    """Generate problems that stress-test common heuristic pitfalls."""
    if seed is not None:
        rng.seed(seed)

    scenario = rng.choice([
        "single_elevator",
        "non_overlapping",
        "tight_capacity",
        "transfer_required",
        "person_at_goal",
        "many_persons_one_elevator",
        "all_same_floor",
        "only_two_floors",
        "person_unreachable_direct",
        "heavy_persons",
    ])

    height = rng.randint(4, 7)

    if scenario == "single_elevator":
        floors = tuple(range(0, height + 1))
        elev = {0: (rng.choice(floors), floors, rng.choice([8, 10, 15]))}
        persons = {}
        for i in range(rng.randint(1, 3)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(1, 5), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "non_overlapping":
        mid = height // 2
        e0 = tuple(range(0, mid + 1))
        e1 = tuple(range(mid, height + 1))
        elev = {0: (rng.choice(e0), e0, 12), 1: (rng.choice(e1), e1, 12)}
        persons = {}
        for i in range(rng.randint(1, 3)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(1, 4), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "tight_capacity":
        floors = tuple(range(0, height + 1))
        elev = {0: (0, floors, 4), 1: (height, floors, 4)}
        persons = {}
        for i in range(rng.randint(2, 3)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(2, 4), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "transfer_required":
        mid = height // 2
        overlap = rng.randint(1, 2)
        e0 = tuple(range(0, mid + overlap + 1))
        e1 = tuple(range(mid, height + 1))
        elev = {0: (0, e0, 10), 1: (height, e1, 10)}
        lo_start = rng.randint(0, max(0, mid - 1))
        hi_goal = rng.randint(min(height, mid + overlap + 1), height)
        persons = {10: (lo_start, 3, hi_goal)}
        for i in range(rng.randint(0, 2)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[11 + i] = (s, rng.randint(1, 4), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "person_at_goal":
        floors = tuple(range(0, height + 1))
        elev = {0: (0, floors, 15)}
        persons = {10: (3, 3, 3), 11: (0, 4, height)}
        if rng.random() < 0.5:
            persons[12] = (height, 2, height)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "many_persons_one_elevator":
        floors = tuple(range(0, height + 1))
        elev = {0: (0, floors, rng.choice([6, 8]))}
        persons = {}
        for i in range(rng.randint(2, 4)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(1, 3), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "all_same_floor":
        floors = tuple(range(0, height + 1))
        f = rng.randint(0, height)
        elev = {0: (f, floors, 15)}
        persons = {}
        for i in range(rng.randint(2, 3)):
            persons[10 + i] = (f, rng.randint(1, 4), rng.randint(0, height))
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "only_two_floors":
        elev = {0: (0, (0, height), 10)}
        persons = {10: (0, 3, height), 11: (height, 3, 0)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "person_unreachable_direct":
        mid = height // 2
        floors0 = tuple(range(0, height + 1))
        elev = {0: (0, floors0, 10)}
        persons = {10: (mid, 3, height)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "heavy_persons":
        floors = tuple(range(0, height + 1))
        elev = {0: (0, floors, 6), 1: (height, floors, 6)}
        persons = {}
        for i in range(rng.randint(2, 3)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(4, 6), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    return generate_random_problem(seed)


def generate_adversarial_problem(seed=None):
    """
    Adversarial cases designed to catch specific heuristic overestimation bugs:

      1. person_already_delivered    – person starts and ends on same floor
      2. elevator_at_person          – elevator is already at person's floor
      3. elevator_at_goal            – elevator is already at person's goal floor
      4. zero_distance_move          – elevator needs to move 0 floors (already optimal)
      5. chain_transfer              – person must transfer through 2 intermediate elevators
      6. one_floor_range             – elevator serves only a single floor (degenerate)
      7. multiple_goals_same_floor   – several persons share the same goal floor
      8. elevator_overloaded_bypass  – some elevators cannot carry the person (too heavy)
      9. persons_swap_floors         – two persons need to swap floors (both use same elevator)
     10. max_weight_exactly_fits     – person weight == elevator capacity (tight)
     11. single_person_many_elevs    – one person, many elevators, only one can serve them
     12. all_persons_at_goal         – trivially solved problem (h must return 0)
     13. tall_building_sparse_stops  – tall building, elevator with widely spaced stops
     14. two_elevators_same_range    – redundant elevators, both can solve the problem
     15. person_needs_two_trips      – capacity forces two separate trips
    """
    if seed is not None:
        rng.seed(seed)

    scenario = rng.choice([
        "person_already_delivered",
        "elevator_at_person",
        "elevator_at_goal",
        "zero_distance_move",
        "chain_transfer",
        "one_floor_range",
        "multiple_goals_same_floor",
        "elevator_overloaded_bypass",
        "persons_swap_floors",
        "max_weight_exactly_fits",
        "single_person_many_elevs",
        "all_persons_at_goal",
        "tall_building_sparse_stops",
        "two_elevators_same_range",
        "person_needs_two_trips",
    ])

    height = rng.randint(4, 8)

    if scenario == "person_already_delivered":
        # All persons are already at their goal (h* = 0)
        floors = tuple(range(0, height + 1))
        elev = {0: (rng.randint(0, height), floors, 15)}
        persons = {}
        for i in range(rng.randint(1, 4)):
            f = rng.randint(0, height)
            persons[10 + i] = (f, rng.randint(1, 5), f)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "elevator_at_person":
        # Elevator is already sitting at the person's floor – no pickup travel needed
        floors = tuple(range(0, height + 1))
        pf = rng.randint(0, height)
        gf = rng.randint(0, height)
        elev = {0: (pf, floors, 10)}
        persons = {10: (pf, 4, gf)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "elevator_at_goal":
        # Elevator is already at person's goal floor
        floors = tuple(range(0, height + 1))
        pf = rng.randint(0, height)
        gf = rng.randint(0, height)
        ef = gf
        elev = {0: (ef, floors, 10)}
        persons = {10: (pf, 4, gf)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "zero_distance_move":
        # Person is already at goal; elevator is there too
        floors = tuple(range(0, height + 1))
        f = rng.randint(0, height)
        elev = {0: (f, floors, 15)}
        extra = {}
        for i in range(rng.randint(0, 2)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            extra[11 + i] = (s, rng.randint(1, 4), g)
        persons = {10: (f, 3, f), **extra}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "chain_transfer":
        # Three elevators each covering one third of the building.
        # Person must transfer twice to get from bottom to top.
        third = height // 3
        e0 = tuple(range(0, third + 2))
        e1 = tuple(range(third, 2 * third + 2))
        e2 = tuple(range(2 * third, height + 1))
        elev = {
            0: (0, e0, 10),
            1: (third, e1, 10),
            2: (2 * third, e2, 10),
        }
        persons = {10: (0, 3, height)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "one_floor_range":
        # Degenerate elevator that only serves a single floor — useless for transport
        floors = tuple(range(0, height + 1))
        f = rng.randint(1, height - 1)
        elev = {
            0: (f, (f,), 10),          # single-floor elevator (can't move)
            1: (0, floors, 10),         # real elevator
        }
        persons = {10: (rng.randint(0, height), 3, rng.randint(0, height))}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "multiple_goals_same_floor":
        # Many persons all want to reach the same goal floor
        floors = tuple(range(0, height + 1))
        goal = rng.randint(1, height)
        elev = {0: (0, floors, 20)}
        persons = {}
        for i in range(rng.randint(2, 4)):
            s = rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(1, 4), goal)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "elevator_overloaded_bypass":
        # Elevator 0 can't carry the heavy person; elevator 1 can
        floors = tuple(range(0, height + 1))
        weight = rng.randint(7, 10)
        elev = {
            0: (0, floors, weight - 1),   # too weak
            1: (height, floors, weight + 5),  # strong enough
        }
        pf = rng.randint(0, height)
        gf = rng.randint(0, height)
        persons = {10: (pf, weight, gf)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "persons_swap_floors":
        # Two persons need to swap floors; they can't both be in the elevator at once
        floors = tuple(range(0, height + 1))
        fa = rng.randint(0, height // 2)
        fb = rng.randint(height // 2 + 1, height)
        cap = rng.choice([3, 4, 5])
        elev = {0: (fa, floors, cap)}
        persons = {
            10: (fa, cap - 1, fb),
            11: (fb, cap - 1, fa),
        }
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "max_weight_exactly_fits":
        # Person's weight exactly matches the elevator capacity
        floors = tuple(range(0, height + 1))
        weight = rng.randint(3, 10)
        elev = {0: (0, floors, weight)}
        pf = rng.randint(0, height)
        gf = rng.randint(0, height)
        persons = {10: (pf, weight, gf)}
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "single_person_many_elevs":
        # Many elevators but only one can reach the person's goal
        height = max(height, 5)
        goal = height
        # Only elevator 0 can reach `goal`
        e0_floors = tuple(range(0, height + 1))
        elevators = {0: (0, e0_floors, 10)}
        for i in range(1, rng.randint(2, 4)):
            partial = tuple(range(0, height - 1))  # can't reach `goal`
            elevators[i] = (0, partial, 10)
        pf = rng.randint(0, height - 1)
        persons = {10: (pf, 3, goal)}
        return {"height": height, "Elevators": elevators, "Persons": persons}

    elif scenario == "all_persons_at_goal":
        # Trivially already solved — heuristic must return 0 from initial state
        floors = tuple(range(0, height + 1))
        elev = {0: (rng.randint(0, height), floors, 15)}
        persons = {}
        for i in range(rng.randint(1, 4)):
            f = rng.randint(0, height)
            persons[10 + i] = (f, rng.randint(1, 5), f)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "tall_building_sparse_stops":
        # Tall building, elevator only stops at even floors
        height = rng.choice([8, 10])
        even_floors = tuple(range(0, height + 1, 2))
        elev = {0: (0, even_floors, 15)}
        pf = rng.choice(even_floors)
        gf = rng.choice(even_floors)
        persons = {10: (pf, 5, gf)}
        for i in range(rng.randint(0, 2)):
            sf = rng.choice(even_floors)
            gff = rng.choice(even_floors)
            persons[11 + i] = (sf, rng.randint(1, 4), gff)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "two_elevators_same_range":
        # Both elevators cover the whole building – one is sufficient but both available
        floors = tuple(range(0, height + 1))
        elev = {
            0: (0, floors, rng.choice([8, 10])),
            1: (height, floors, rng.choice([8, 10])),
        }
        persons = {}
        for i in range(rng.randint(1, 3)):
            s, g = rng.randint(0, height), rng.randint(0, height)
            persons[10 + i] = (s, rng.randint(1, 5), g)
        return {"height": height, "Elevators": elev, "Persons": persons}

    elif scenario == "person_needs_two_trips":
        # Total weight exceeds capacity; persons must be transported separately
        floors = tuple(range(0, height + 1))
        cap = rng.randint(3, 5)
        elev = {0: (0, floors, cap)}
        goal = rng.randint(1, height)
        persons = {}
        for i in range(rng.randint(2, 3)):
            pf = 0
            persons[10 + i] = (pf, cap - 1, goal)
        return {"height": height, "Elevators": elev, "Persons": persons}

    return generate_random_problem(seed)


# ═════════════════════════════════════════════════════════════════════════════
#  SOLVERS
# ═════════════════════════════════════════════════════════════════════════════

def solve_ucs(problem):
    try:
        return search.astar_search(problem, h=lambda n: 0)
    except Exception:
        return None


def solve_astar(problem):
    try:
        return search.astar_search(problem, h=problem.h_astar)
    except Exception:
        return None


def extract_path(goal_node):
    return goal_node.path()[::-1]


# ═════════════════════════════════════════════════════════════════════════════
#  EXHAUSTIVE h* (backward BFS over full state space)
# ═════════════════════════════════════════════════════════════════════════════

def compute_true_hstar(problem, max_states=50000):
    """
    Forward-BFS the entire reachable state space from initial, then
    multi-source BFS *backward* from all goal states.

    Every action costs 1, so BFS gives optimal distances in O(|S|+|E|).

    Returns { state: h*(state) } or None if the forward state space
    exceeds max_states.
    """
    initial = problem.initial
    backward = {}
    queue = deque([initial])
    visited = {initial}
    goals = []

    while queue:
        if len(visited) > max_states:
            return None

        state = queue.popleft()

        if problem.goal_test(state):
            goals.append(state)

        try:
            successors = problem.successor(state)
        except Exception:
            continue

        for action, next_state in successors:
            preds = backward.get(next_state)
            if preds is None:
                backward[next_state] = [state]
            else:
                preds.append(state)

            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    if not goals:
        return None

    hstar = {g: 0 for g in goals}
    bfs_queue = deque(goals)
    while bfs_queue:
        state = bfs_queue.popleft()
        d_next = hstar[state] + 1
        preds = backward.get(state)
        if preds is None:
            continue
        for prev_state in preds:
            if prev_state not in hstar:
                hstar[prev_state] = d_next
                bfs_queue.append(prev_state)

    return hstar


# ═════════════════════════════════════════════════════════════════════════════
#  KNOWN PROBLEMS
# ═════════════════════════════════════════════════════════════════════════════

KNOWN_PROBLEMS = [
    ("p1", {
        "height": 6,
        "Elevators": {0:(0,(0,1,2,3),8), 1:(4,(2,4,5,6),10)},
        "Persons": {10:(0,3,3), 11:(2,4,6), 12:(4,5,0)}
    }, 13),
    ("e1", {
        "height": 5,
        "Elevators": {0:(0,(0,1,2,3,4,5),15), 1:(5,(0,1,2,3,4,5),15)},
        "Persons": {10:(0,3,5), 11:(5,3,0), 12:(3,3,1)}
    }, 10),
    ("e2", {
        "height": 6,
        "Elevators": {0:(0,(0,1,2,3),10), 1:(6,(3,4,5,6),10)},
        "Persons": {10:(1,3,3), 11:(5,3,4), 12:(0,3,2)}
    }, 11),
    ("e3", {
        "height": 6,
        "Elevators": {0:(0,(0,1,2,3,4),10), 1:(6,(2,4,5,6),10)},
        "Persons": {10:(0,3,5), 11:(6,3,1), 12:(3,4,6)}
    }, 18),
    ("e4", {
        "height": 5,
        "Elevators": {0:(0,(0,1,2,3,4,5),7), 1:(5,(0,1,2,3,4,5),7)},
        "Persons": {10:(0,5,5), 11:(0,5,3), 12:(5,5,0)}
    }, 9),
    ("e5", {
        "height": 7,
        "Elevators": {0:(0,(0,1,2,3,4),12), 1:(7,(3,4,5,6,7),12)},
        "Persons": {10:(1,4,3), 11:(6,4,7), 12:(0,4,7)}
    }, 13),
    ("m1", {
        "height": 6,
        "Elevators": {0:(0,(0,1,2,3),12), 1:(6,(3,4,5,6),12)},
        "Persons": {10:(0,4,5), 11:(5,4,0), 12:(2,4,6), 13:(5,4,1)}
    }, 24),
    ("m2", {
        "height": 8,
        "Elevators": {0:(0,(0,1,2,3,4),10), 1:(4,(2,4,6,8),10)},
        "Persons": {10:(0,3,8), 11:(8,3,0), 12:(2,3,6), 13:(6,3,1)}
    }, 21),
    ("m3", {
        "height": 8,
        "Elevators": {0:(0,(0,1,2,3,4),10), 1:(4,(2,4,6,8),10), 2:(4,(7,2),10)},
        "Persons": {10:(0,3,8), 11:(8,3,0), 12:(2,3,7), 13:(6,3,1)}
    }, 22),
    ("m4", {
        "height": 6,
        "Elevators": {0:(0,(0,1,2,3,4,5,6),8), 1:(6,(0,1,2,3,4,5,6),8)},
        "Persons": {10:(0,5,6), 11:(0,5,4), 12:(6,5,0), 13:(6,5,2), 14:(3,5,6)}
    }, 16),
    ("m5", {
        "height": 8,
        "Elevators": {0:(0,(0,1,2,3,4),10), 1:(4,(4,5,6,7,8),10)},
        "Persons": {10:(0,6,8), 11:(0,4,5), 12:(8,6,0), 13:(8,5,3)}
    }, 25),
]


# ═════════════════════════════════════════════════════════════════════════════
#  TEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class Stats:
    def __init__(self):
        self.total = 0
        self.skipped_unsolvable = 0
        self.skipped_timeout = 0
        self.skipped_error = 0
        self.optimality_pass = 0
        self.optimality_fail = 0
        self.path_adm_pass = 0
        self.path_adm_fail = 0
        self.consistency_pass = 0
        self.consistency_fail = 0
        self.hgoal_pass = 0
        self.hgoal_fail = 0
        self.exhaustive_tested = 0
        self.exhaustive_pass = 0
        self.exhaustive_fail = 0
        self.exhaustive_states_checked = 0
        self.failures = []


def test_single(name, init_dict, known_opt, stats, do_exhaustive=True,
                verbose=True, max_states_exhaustive=50000, time_limit=30.0):
    """Run all admissibility checks on one problem instance."""
    stats.total += 1
    t_start = time.time()

    try:
        prob_ucs = ex1.create_elevators_problem(init_dict)
    except Exception as e:
        if verbose:
            print(f"  [{name}] ERROR creating problem: {e}")
        stats.skipped_error += 1
        return

    try:
        ucs_result = solve_ucs(prob_ucs)
    except Exception as e:
        if verbose:
            print(f"  [{name}] UCS exception: {e}")
        stats.skipped_error += 1
        return

    if ucs_result is None or not isinstance(ucs_result[0], Node):
        stats.skipped_unsolvable += 1
        return

    opt_cost = ucs_result[0].path_cost

    if known_opt is not None and opt_cost != known_opt:
        if verbose:
            print(f"  [{name}] ⚠ UCS={opt_cost} != expected={known_opt}")

    if time.time() - t_start > time_limit:
        stats.skipped_timeout += 1
        return

    # ── Solve with A* ──
    try:
        prob_astar = ex1.create_elevators_problem(init_dict)
        astar_result = solve_astar(prob_astar)
    except Exception as e:
        if verbose:
            print(f"  [{name}] A* exception: {e}")
        stats.skipped_error += 1
        return

    if astar_result is None or not isinstance(astar_result[0], Node):
        msg = f"A* found no solution; UCS optimal={opt_cost}"
        if verbose:
            print(f"  [{name}] ✗ {msg}")
        stats.optimality_fail += 1
        stats.failures.append((name, "NO_SOLUTION", msg, init_dict))
        return

    astar_node = astar_result[0]
    astar_cost = astar_node.path_cost

    # ── CHECK 1: Optimality ──
    if astar_cost != opt_cost:
        direction = "OVERESTIMATE" if astar_cost > opt_cost else "BUG(under)"
        msg = f"A*={astar_cost} UCS={opt_cost} [{direction}]"
        if verbose:
            print(f"  [{name}] ✗ OPTIMALITY: {msg}")
        stats.optimality_fail += 1
        stats.failures.append((name, "OPTIMALITY", msg, init_dict))
        return
    stats.optimality_pass += 1

    # ── CHECK 2: h along optimal path ──
    path_nodes = extract_path(astar_node)
    path_ok = True
    for node in path_nodes:
        remaining = opt_cost - node.path_cost
        h_val = prob_astar.h_astar(node)
        if h_val > remaining:
            msg = (f"depth={node.depth} g={node.path_cost} "
                   f"h={h_val} remaining={remaining} Δ={h_val-remaining}")
            if verbose:
                print(f"  [{name}] ✗ PATH_ADM: {msg}")
            stats.path_adm_fail += 1
            stats.failures.append((name, "PATH_ADM", msg, init_dict))
            path_ok = False
            break
    if path_ok:
        stats.path_adm_pass += 1

    # ── CHECK 3: Consistency along path ──
    cons_ok = True
    for i in range(len(path_nodes) - 1):
        n, n2 = path_nodes[i], path_nodes[i + 1]
        h_n = prob_astar.h_astar(n)
        h_n2 = prob_astar.h_astar(n2)
        c = n2.path_cost - n.path_cost
        if h_n > c + h_n2:
            msg = (f"depth {n.depth}→{n2.depth}: h={h_n} > "
                   f"c={c}+h'={h_n2}={c+h_n2}")
            if verbose:
                print(f"  [{name}] ✗ CONSISTENCY: {msg}")
            stats.consistency_fail += 1
            stats.failures.append((name, "CONSISTENCY", msg, init_dict))
            cons_ok = False
            break
    if cons_ok:
        stats.consistency_pass += 1

    # ── CHECK 4: h(goal)==0 ──
    goal_h = prob_astar.h_astar(astar_node)
    if goal_h != 0:
        msg = f"h(goal)={goal_h}"
        if verbose:
            print(f"  [{name}] ✗ H_GOAL: {msg}")
        stats.hgoal_fail += 1
        stats.failures.append((name, "H_GOAL", msg, init_dict))
    else:
        stats.hgoal_pass += 1

    # ── CHECK 5: Exhaustive h* on full state space ──
    if not do_exhaustive:
        return
    if time.time() - t_start > time_limit * 0.8:
        return

    try:
        prob_exh = ex1.create_elevators_problem(init_dict)
        hstar = compute_true_hstar(prob_exh, max_states=max_states_exhaustive)
    except Exception:
        return

    if hstar is None:
        return

    stats.exhaustive_tested += 1
    num_states = len(hstar)
    stats.exhaustive_states_checked += num_states

    prob_check = ex1.create_elevators_problem(init_dict)
    violations = 0
    worst_overest = 0
    worst_state = None
    worst_hval = 0
    worst_true = 0

    for state, true_h in hstar.items():
        dummy = Node(state)
        dummy.path_cost = 0
        try:
            h_val = prob_check.h_astar(dummy)
        except Exception:
            continue

        if h_val > true_h:
            violations += 1
            overest = h_val - true_h
            if overest > worst_overest:
                worst_overest = overest
                worst_state = state
                worst_hval = h_val
                worst_true = true_h

    if violations == 0:
        stats.exhaustive_pass += 1
        if verbose:
            print(f"  [{name}] ✓ Exhaustive OK: {num_states} states, opt={opt_cost}")
    else:
        stats.exhaustive_fail += 1
        msg = (f"{violations}/{num_states} states overestimate. "
               f"Worst: h={worst_hval} vs h*={worst_true} (Δ={worst_overest})")
        if verbose:
            print(f"  [{name}] ✗ EXHAUSTIVE: {msg}")
            print(f"           State: {worst_state}")
        stats.failures.append((name, "EXHAUSTIVE", msg, init_dict))


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    stats = Stats()
    wall_start = time.time()

    # ── Phase 1: Known problems ──────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: Known problems (11 instances, exhaustive h* check)")
    print("=" * 70)

    for name, init_dict, known_opt in KNOWN_PROBLEMS:
        t0 = time.time()
        test_single(name, init_dict, known_opt, stats,
                    do_exhaustive=True, verbose=True,
                    max_states_exhaustive=200000)
        dt = time.time() - t0
        if dt > 1.0:
            print(f"    ({dt:.1f}s)")

    # ── Phase 2: 3000 random problems ────────────────────────────────────────
    N2 = 3000
    print()
    print("=" * 70)
    print(f"PHASE 2: {N2} random problems")
    print("=" * 70)

    p2_fails = 0
    for i in range(N2):
        d = generate_random_problem(seed=i)
        old = len(stats.failures)
        test_single(f"rand_{i:05d}", d, None, stats,
                    do_exhaustive=True, verbose=False,
                    max_states_exhaustive=40000)
        if len(stats.failures) > old:
            p2_fails += 1
            for f in stats.failures[old:]:
                print(f"  ✗ [{f[0]}] {f[1]}: {f[2]}")
                print(f"    {f[3]}")
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{N2}  "
                  f"(unsolvable={stats.skipped_unsolvable}, "
                  f"fails={p2_fails})")

    # ── Phase 3: 3000 targeted edge-case problems ─────────────────────────────
    N3 = 3000
    print()
    print("=" * 70)
    print(f"PHASE 3: {N3} targeted edge-case problems")
    print("=" * 70)

    p3_fails = 0
    for i in range(N3):
        d = generate_targeted_problem(seed=10000 + i)
        old = len(stats.failures)
        test_single(f"tgt_{i:05d}", d, None, stats,
                    do_exhaustive=True, verbose=False,
                    max_states_exhaustive=40000)
        if len(stats.failures) > old:
            p3_fails += 1
            for f in stats.failures[old:]:
                print(f"  ✗ [{f[0]}] {f[1]}: {f[2]}")
                print(f"    {f[3]}")
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{N3}  "
                  f"(unsolvable={stats.skipped_unsolvable}, "
                  f"fails={p3_fails})")

    # ── Phase 4: 2000 stress tests ────────────────────────────────────────────
    N4 = 2000
    print()
    print("=" * 70)
    print(f"PHASE 4: {N4} larger stress-test problems (4 persons, 2-3 elevators)")
    print("=" * 70)

    p4_fails = 0
    for i in range(N4):
        rng.seed(20000 + i)
        height = rng.randint(5, 8)
        ne = rng.randint(2, 3)
        np_ = rng.randint(3, 4)
        elevators = {}
        for eid in range(ne):
            if rng.random() < 0.5:
                fl = random_contiguous_floors(height)
            else:
                fl = random_sparse_floors(height, 2)
            elevators[eid] = (rng.choice(fl), fl, rng.choice([6, 8, 10, 12]))
        persons = {}
        for pid in range(10, 10 + np_):
            persons[pid] = (rng.randint(0, height),
                            rng.randint(1, 5),
                            rng.randint(0, height))
        d = {"height": height, "Elevators": elevators, "Persons": persons}

        old = len(stats.failures)
        test_single(f"stress_{i:05d}", d, None, stats,
                    do_exhaustive=True, verbose=False,
                    max_states_exhaustive=30000,
                    time_limit=15.0)
        if len(stats.failures) > old:
            p4_fails += 1
            for f in stats.failures[old:]:
                print(f"  ✗ [{f[0]}] {f[1]}: {f[2]}")
                print(f"    {f[3]}")
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{N4}  "
                  f"(unsolvable={stats.skipped_unsolvable}, "
                  f"fails={p4_fails})")

    # ── Phase 5: 2000 adversarial corner-case problems ────────────────────────
    N5 = 2000
    print()
    print("=" * 70)
    print(f"PHASE 5: {N5} adversarial corner-case problems (15 targeted scenarios)")
    print("=" * 70)

    p5_fails = 0
    scenario_counts = Counter()
    for i in range(N5):
        d = generate_adversarial_problem(seed=30000 + i)
        old = len(stats.failures)
        test_single(f"adv_{i:05d}", d, None, stats,
                    do_exhaustive=True, verbose=False,
                    max_states_exhaustive=40000,
                    time_limit=20.0)
        if len(stats.failures) > old:
            p5_fails += 1
            for f in stats.failures[old:]:
                print(f"  ✗ [{f[0]}] {f[1]}: {f[2]}")
                print(f"    {f[3]}")
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{N5}  "
                  f"(unsolvable={stats.skipped_unsolvable}, "
                  f"fails={p5_fails})")

    # ── Final Report ──────────────────────────────────────────────────────────
    wall_time = time.time() - wall_start
    tested = stats.optimality_pass + stats.optimality_fail

    print()
    print("=" * 70)
    print("                        FINAL REPORT")
    print("=" * 70)
    print(f"  Instances generated    : {stats.total}")
    print(f"  Skipped (unsolvable)   : {stats.skipped_unsolvable}")
    print(f"  Skipped (error)        : {stats.skipped_error}")
    print(f"  Skipped (timeout)      : {stats.skipped_timeout}")
    print(f"  Actually tested        : {tested}")
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  Optimality      PASS {stats.optimality_pass:>5}  FAIL {stats.optimality_fail:>5}  │")
    print(f"  │  Path admiss.    PASS {stats.path_adm_pass:>5}  FAIL {stats.path_adm_fail:>5}  │")
    print(f"  │  Consistency     PASS {stats.consistency_pass:>5}  FAIL {stats.consistency_fail:>5}  │")
    print(f"  │  h(goal)==0      PASS {stats.hgoal_pass:>5}  FAIL {stats.hgoal_fail:>5}  │")
    print(f"  │                                             │")
    print(f"  │  Exhaustive h*   tested  {stats.exhaustive_tested:>5}               │")
    print(f"  │  Exhaustive h*   PASS {stats.exhaustive_pass:>5}  FAIL {stats.exhaustive_fail:>5}  │")
    print(f"  │  States verified (h*)  {stats.exhaustive_states_checked:>9}           │")
    print(f"  └─────────────────────────────────────────────┘")
    print()
    print(f"  Wall time: {wall_time:.1f}s")

    if stats.failures:
        print()
        print(f"  ╔═══════════════════════════════════════════════╗")
        print(f"  ║  {len(stats.failures):>3} FAILURE(S) DETECTED                      ║")
        print(f"  ╚═══════════════════════════════════════════════╝")
        print()
        by_type = Counter(f[1] for f in stats.failures)
        for ftype, count in by_type.most_common():
            print(f"    {ftype:>15}: {count}")
        print()
        print("  First 20 failures (with reproducible problem dicts):")
        print("  " + "-" * 66)
        for name, ftype, detail, pdict in stats.failures[:20]:
            print(f"    [{name}] {ftype}: {detail}")
            print(f"      {pdict}")
            print()
    else:
        print()
        print(f"  ╔═══════════════════════════════════════════════╗")
        print(f"  ║       ✓✓✓  ALL TESTS PASSED  ✓✓✓             ║")
        print(f"  ╚═══════════════════════════════════════════════╝")
        print()
        print(f"  Heuristic appears admissible & consistent across")
        print(f"  {tested} instances with {stats.exhaustive_states_checked:,} "
              f"states verified exhaustively.")

    print("=" * 70)
    return len(stats.failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
