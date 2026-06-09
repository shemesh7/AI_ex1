import ex1_check
import search as search
import utils as utils
from collections import deque

# AI: Used Claude Sonnet 4.6 for implementing some of my ideas, code review and improvements.
id = ["331050591"]

INF = float('inf')


# ── Problem ───────────────────────────────────────────────────────────────────
class ElevatorsProblem(search.Problem):
    """
    State = flat tuple  (e_floor_0, …, e_floor_{n_e-1},
                          p_loc_0,   …, p_loc_{n_p-1})

    Person location encoding:
        loc <  ELEV_OFFSET  →  standing on floor `loc`
        loc >= ELEV_OFFSET  →  inside elevator at index (loc − ELEV_OFFSET)

    Using (height+1) as ELEV_OFFSET separates floor indices from elevator
    indices with no arithmetic overhead and lets h_costs[j][loc] be
    indexed directly without any conversion.
    """

    def __init__(self, initial):
        height = initial['height']
        self.height = height
        self.e_ids = tuple(sorted(initial['Elevators'].keys()))
        self.p_ids = tuple(sorted(initial['Persons'].keys()))
        n_e = len(self.e_ids)
        n_p = len(self.p_ids)
        self.n_e = n_e
        self.n_p = n_p

        ELEV_OFFSET = height + 1        # floor indices: 0..height < ELEV_OFFSET
        self.ELEV_OFFSET = ELEV_OFFSET
        num_nodes = ELEV_OFFSET + n_e   # total nodes in the bipartite graph

        # ── Static elevator data ──────────────────────────────────────────
        e_reach = [list(initial['Elevators'][eid][1]) for eid in self.e_ids]
        self.e_capacity = tuple(initial['Elevators'][eid][2] for eid in self.e_ids)
        e_init_floors = tuple(initial['Elevators'][eid][0] for eid in self.e_ids)
        e_reach_sets  = [frozenset(r) for r in e_reach]

        # ── Static person data ────────────────────────────────────────────
        self.p_weights = tuple(initial['Persons'][pid][1] for pid in self.p_ids)
        self.p_goals   = tuple(initial['Persons'][pid][2] for pid in self.p_ids)
        p_starts       = tuple(initial['Persons'][pid][0] for pid in self.p_ids)

        # ── BFS: h_costs[j][node] = min (ENTER+EXIT) for person j ────────
        # Bipartite graph: floor nodes (0..height) ↔ elevator nodes
        # (ELEV_OFFSET+i). Each edge costs 1 (one ENTER or EXIT). BFS from
        # goal_floor backward gives the minimum enter/exit count from any
        # node. Elevators that cannot carry person j are excluded, so the
        # bound is weight-aware and tighter than a single shared BFS.
        h_costs = []
        for j in range(n_p):
            pw     = self.p_weights[j]
            goal_f = self.p_goals[j]
            adj    = [[] for _ in range(num_nodes)]
            for i in range(n_e):
                if self.e_capacity[i] < pw:
                    continue
                e_node = ELEV_OFFSET + i
                for f in e_reach[i]:
                    adj[f].append(e_node)
                    adj[e_node].append(f)
                # Virtual edge: initial floor not in normal reach (chain elevators).
                # Makes h_costs finite so A* can find a path; heuristic still
                # admissible (underestimates at worst).
                init_f = e_init_floors[i]
                if init_f not in e_reach_sets[i]:
                    adj[init_f].append(e_node)
                    adj[e_node].append(init_f)
            dist = [INF] * num_nodes
            dist[goal_f] = 0
            q = deque([goal_f])
            while q:
                curr = q.popleft()
                nd = dist[curr] + 1
                for nb in adj[curr]:
                    if dist[nb] == INF:
                        dist[nb] = nd
                        q.append(nb)
            h_costs.append(tuple(dist))
        self.h_costs = tuple(h_costs)

        # ── Bitmask structures for fast successor generation ──────────────

        # e_reach_mask[i] = bitmask of all floors reachable by elevator i
        e_reach_mask = [0] * n_e
        for i in range(n_e):
            for f in e_reach[i]:
                if 0 <= f <= height:
                    e_reach_mask[i] |= 1 << f
        self.e_reach_mask = tuple(e_reach_mask)

        # source_covers[j][f] = bitmask of elevator indices i such that:
        #   - elevator i can reach floor f
        #   - h_costs[j][e_node_i] == h_costs[j][f] - 1  (i is on BFS path)
        # Used in MOVE (elevator should come to floor f) and ENTER (only
        # enter an elevator that is on the optimal BFS path).
        source_covers = []
        for j in range(n_p):
            row = [0] * (height + 1)
            for f in range(height + 1):
                hf = h_costs[j][f]
                if hf == INF or hf == 0:
                    continue
                target = hf - 1
                for i in range(n_e):
                    if (e_reach_mask[i] >> f) & 1 and h_costs[j][ELEV_OFFSET + i] == target:
                        row[f] |= 1 << i
            source_covers.append(tuple(row))
        self.source_covers = tuple(source_covers)

        # exit_floors_mask[j][i] = bitmask of floors where person j should
        # exit elevator i on the BFS optimal path:
        #   h_costs[j][f] == h_costs[j][e_node_i] - 1
        # Restricting exits to this set prunes all suboptimal EXIT actions.
        exit_floors_mask = []
        for j in range(n_p):
            row = [0] * n_e
            for i in range(n_e):
                d = h_costs[j][ELEV_OFFSET + i]
                if d == INF or d == 0:
                    continue
                target = d - 1
                for f in e_reach[i]:
                    if 0 <= f <= height and h_costs[j][f] == target:
                        row[i] |= 1 << f
            exit_floors_mask.append(tuple(row))
        self.exit_floors_mask = tuple(exit_floors_mask)

        # LSB_MAP: power-of-2 → exponent, for fast bitmask iteration
        max_bits = max(height + 1, n_e, n_p) + 1
        self.LSB_MAP = {1 << k: k for k in range(max_bits)}

        # ── Precomputed action strings ────────────────────────────────────
        # Building f-strings inside the successor hot path is measurably slow.
        move_str = []
        for i in range(n_e):
            row = [None] * (height + 1)
            for f in e_reach[i]:
                row[f] = "MOVE{%d,%d}" % (self.e_ids[i], f)
            move_str.append(tuple(row))
        self.move_str  = tuple(move_str)
        self.enter_str = tuple(
            tuple("ENTER{%d,%d}" % (self.p_ids[j], self.e_ids[i])
                  for i in range(n_e))
            for j in range(n_p)
        )
        self.exit_str = tuple(
            tuple("EXIT{%d,%d}" % (self.p_ids[j], self.e_ids[i])
                  for i in range(n_e))
            for j in range(n_p)
        )

        # ── Initial state ─────────────────────────────────────────────────
        search.Problem.__init__(self, e_init_floors + p_starts)

    # ── successor ─────────────────────────────────────────────────────────────
    def successor(self, state):
        """
        Pruning rules (each preserves at least one optimal plan):

          MOVE  – dynamic allowed_move_mask[i]: only floors that elevator i
                  actually needs to reach in this state:
                  • exit floors for current passengers (on BFS optimal path)
                  • floor of each waiting person whose BFS path goes through i
                  This is far tighter than a static source/goal/transfer set.

          EXIT  – only at floors in exit_floors_mask[j][i] (BFS-optimal exits).
                  Exiting at a non-optimal floor raises h_costs and is never
                  part of an optimal plan.
                  Special case: if person is at their goal inside elevator,
                  force immediate exit (return only that action).

          ENTER – only onto elevators in source_covers[j][loc] (BFS-path
                  elevators). Entering a non-path elevator raises h_costs.
                  Also skip if over capacity.
        """
        n_e = self.n_e
        n_p = self.n_p
        ELEV_OFFSET      = self.ELEV_OFFSET
        e_floors         = state[:n_e]
        p_locs           = state[n_e:]
        p_goals          = self.p_goals
        p_weights        = self.p_weights
        e_capacity       = self.e_capacity
        source_covers    = self.source_covers
        exit_floors_mask = self.exit_floors_mask
        e_reach_mask     = self.e_reach_mask
        LSB_MAP          = self.LSB_MAP
        move_str         = self.move_str
        enter_str        = self.enter_str
        exit_str         = self.exit_str

        h_costs           = self.h_costs
        e_weights         = [0] * n_e
        allowed_move_mask = [0] * n_e
        successors        = []

        for j in range(n_p):
            loc = p_locs[j]
            g   = p_goals[j]
            if loc >= ELEV_OFFSET:
                ei = loc - ELEV_OFFSET
                f  = e_floors[ei]
                if f == g:
                    # Person is at goal inside elevator — force immediate exit.
                    # Delaying is never beneficial with unit costs.
                    idx = n_e + j
                    return [(exit_str[j][ei], state[:idx] + (f,) + state[idx + 1:])]
                e_weights[ei] += p_weights[j]
                allowed_move_mask[ei] |= exit_floors_mask[j][ei]
            elif loc != g:
                cov = source_covers[j][loc]
                if cov:
                    bit = 1 << loc
                    m = cov
                    while m:
                        lsb = m & -m
                        allowed_move_mask[LSB_MAP[lsb]] |= bit
                        m ^= lsb

        # ── MOVE ─────────────────────────────────────────────────────────
        for i in range(n_e):
            mask = allowed_move_mask[i] & e_reach_mask[i] & ~(1 << e_floors[i])
            if not mask:
                continue
            mrow   = move_str[i]
            prefix = state[:i]
            suffix = state[i + 1:]
            m = mask
            while m:
                lsb = m & -m
                target = LSB_MAP[lsb]
                successors.append((mrow[target], prefix + (target,) + suffix))
                m ^= lsb

        # ── EXIT ─────────────────────────────────────────────────────────
        for j in range(n_p):
            loc = p_locs[j]
            if loc < ELEV_OFFSET:
                continue
            ei = loc - ELEV_OFFSET
            f  = e_floors[ei]
            if (exit_floors_mask[j][ei] >> f) & 1:
                idx = n_e + j
                successors.append((exit_str[j][ei], state[:idx] + (f,) + state[idx + 1:]))

        # ── ENTER ────────────────────────────────────────────────────────
        for j in range(n_p):
            loc = p_locs[j]
            if loc >= ELEV_OFFSET or loc == p_goals[j]:
                continue
            pw    = p_weights[j]
            cov_j = source_covers[j][loc]
            m     = cov_j
            while m:
                lsb = m & -m
                i   = LSB_MAP[lsb]
                m  ^= lsb
                if e_floors[i] != loc:
                    continue
                if e_weights[i] + pw > e_capacity[i]:
                    continue
                idx = n_e + j
                successors.append((enter_str[j][i],
                                   state[:idx] + (ELEV_OFFSET + i,) + state[idx + 1:]))
            # Fallback: elevator physically at loc but loc not in its reach
            # (e.g. elevator initialised at a non-stop floor).
            # Allow ENTER when it is still BFS-optimal (h drops by exactly 1).
            h_loc = h_costs[j][loc]
            if h_loc < 2:
                continue
            for i in range(n_e):
                if (cov_j >> i) & 1:
                    continue
                if e_floors[i] != loc:
                    continue
                if (e_reach_mask[i] >> loc) & 1:
                    continue  # in reach — already handled by cov_j
                d_ei = h_costs[j][ELEV_OFFSET + i]
                if d_ei == INF or d_ei != h_loc - 1:
                    continue
                if e_weights[i] + pw > e_capacity[i]:
                    continue
                idx = n_e + j
                successors.append((enter_str[j][i],
                                   state[:idx] + (ELEV_OFFSET + i,) + state[idx + 1:]))

        return successors

    # ── goal_test ─────────────────────────────────────────────────────────────
    def goal_test(self, state):
        return state[self.n_e:] == self.p_goals

    # ── h_astar ───────────────────────────────────────────────────────────────
    def h_astar(self, node):
        """
        h = transfer_lb + strict_move_lb   (admissible + consistent)

        transfer_lb = Σ_j h_costs[j][p_loc_j]
            Each h_costs[j][·] is the BFS min ENTER+EXIT for person j.
            Summing is admissible because every ENTER and EXIT action
            advances exactly one person — no sharing between persons.

        strict_move_lb = #uncovered required floors
            A floor F is *required* if an unfinished person is waiting
            there (source) or must be delivered there (goal).
            F is *covered* iff BOTH:
              (a) src_covered: no waiting person at F, OR some elevator
                  currently at F is on the BFS shortest path for a
                  waiting person [source_covers[j][F] & elevs_at_mask[F]]
              (b) goal_covered: no delivery target for F, OR the target
                  person is already inside an elevator sitting at F.
            AND (not OR) is required: OR would let a single EXIT reduce
            both transfer_lb and strict_move_lb simultaneously, violating
            consistency.

        The two terms bound disjoint action types (ENTER+EXIT vs MOVE),
        so their sum is a valid lower bound on the total remaining cost.
        """
        n_e         = self.n_e
        n_p         = self.n_p
        ELEV_OFFSET = self.ELEV_OFFSET
        height_p1   = self.height + 1
        state       = node.state
        e_floors    = state[:n_e]
        p_locs      = state[n_e:]
        h_costs     = self.h_costs
        p_goals     = self.p_goals
        source_covers = self.source_covers

        h_val = 0
        src_req  = [None] * height_p1
        goal_req = [None] * height_p1
        any_unfinished = False

        for j in range(n_p):
            loc = p_locs[j]
            h_val += h_costs[j][loc]
            g = p_goals[j]
            if loc != g:
                any_unfinished = True
                if loc < ELEV_OFFSET:
                    if src_req[loc] is None:
                        src_req[loc] = [j]
                    else:
                        src_req[loc].append(j)
                if goal_req[g] is None:
                    goal_req[g] = [j]
                else:
                    goal_req[g].append(j)

        if not any_unfinished:
            return h_val

        # Bitmask of elevator indices at each floor (O(1) coverage check).
        elevs_at_mask = [0] * height_p1
        for i in range(n_e):
            f = e_floors[i]
            if 0 <= f < height_p1:
                elevs_at_mask[f] |= 1 << i

        uncovered = 0
        for f in range(height_p1):
            srcs = src_req[f]
            gs   = goal_req[f]
            if srcs is None and gs is None:
                continue

            src_covered  = srcs is None
            goal_covered = gs is None

            if not src_covered:
                eam = elevs_at_mask[f]
                if eam:
                    for j in srcs:
                        if source_covers[j][f] & eam:
                            src_covered = True
                            break

            if not goal_covered:
                for j in gs:
                    loc = p_locs[j]
                    if loc >= ELEV_OFFSET and e_floors[loc - ELEV_OFFSET] == f:
                        goal_covered = True
                        break

            if not (src_covered and goal_covered):
                uncovered += 1

        return h_val + uncovered


def create_elevators_problem(game):
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
