import ex1_check
import search as search
import utils as utils
import heapq
from collections import deque

# AI disclosure: Used Claude Sonnet 4.6 throughout. BFS heuristic, strict_move_lb,
# useful_move_targets, and fast A* driver are adapted from Ophir's approach.
id = ["No numbers - I'm special!"]

INF = float('inf')


# ── Fast A* driver ────────────────────────────────────────────────────────────
def _fast_astar(problem, h=None):
    """A* with:
    - heapq  (O(log n) vs O(n) bisect.insort in the stock PriorityQueue)
    - g_best dedup  (push a child only when it strictly improves the best
                     known g; stale pops are discarded immediately)
    - h tie-break   (among equal-f nodes prefer smaller h — closer to goal)
    - pathmax       (child.f ≥ parent.f restores monotonicity along paths,
                     so with admissible h the first pop of a state is optimal)
    """
    if h is None:
        h = problem.h
    Node = search.Node

    root = Node(problem.initial)
    h0 = h(root)
    root.f = h0

    heap = [(h0, h0, 0, root)]   # (f, h, counter, node)
    g_best = {problem.initial: 0}
    expanded = 0
    ctr = 0
    _pop = heapq.heappop
    _push = heapq.heappush
    succ = problem.successor
    goal = problem.goal_test

    while heap:
        _, _, _, node = _pop(heap)
        s = node.state
        g = node.path_cost
        if g_best.get(s, INF) < g:   # stale entry
            continue
        if goal(s):
            return node, expanded
        expanded += 1
        new_g = g + 1
        pf = node.f
        for action, ns in succ(s):
            if g_best.get(ns, INF) <= new_g:
                continue
            g_best[ns] = new_g
            child = Node(ns, node, action, new_g)
            ch = h(child)
            cf = new_g + ch
            if cf < pf:
                cf = pf          # pathmax
            child.f = cf
            ctr += 1
            _push(heap, (cf, ch, ctr, child))
    return None


search.astar_search = _fast_astar


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

        # elev_useful[j][i] = True iff elevator i can help person j reach goal
        # (BFS distance from elevator-node to goal is finite).
        self.elev_useful = tuple(
            tuple(h_costs[j][ELEV_OFFSET + i] < INF for i in range(n_e))
            for j in range(n_p)
        )

        # ── Useful MOVE targets ───────────────────────────────────────────
        # An elevator never needs to visit a floor that is not:
        #   (a) a start floor of any person,
        #   (b) a goal floor of any person, or
        #   (c) a transfer floor (reachable by ≥ 2 elevators, needed for relay).
        # Any MOVE to a floor outside this set can be deleted from an optimal
        # plan without increasing its cost.
        person_floors = set(p_starts) | set(self.p_goals)
        floor_count   = [0] * (height + 1)
        for i in range(n_e):
            for f in e_reach[i]:
                if 0 <= f <= height:
                    floor_count[f] += 1
        transfer    = {f for f in range(height + 1) if floor_count[f] >= 2}
        useful_glob = person_floors | transfer

        self.useful_targets = tuple(
            tuple(f for f in e_reach[i] if f in useful_glob)
            for i in range(n_e)
        )
        self.useful_sets = tuple(
            frozenset(self.useful_targets[i]) for i in range(n_e)
        )

        # ── Precomputed action strings ─────────────────────────────────
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

        # ── Initial state ─────────────────────────────────────────────
        e_floors = tuple(initial['Elevators'][eid][0] for eid in self.e_ids)
        search.Problem.__init__(self, e_floors + p_starts)

    # ── successor ─────────────────────────────────────────────────────────────
    def successor(self, state):
        """
        Pruning rules (each preserves at least one optimal plan):
          MOVE  – only to useful_targets floors (not start/goal/transfer → useless).
               – skip entirely for an empty elevator when no waiting person
                 is at any of its useful targets.
          EXIT  – skip if h_costs[j][exit_floor] == INF (person can never
                  reach their goal from that floor; no optimal plan uses it).
          ENTER – skip if person is already at goal floor.
               – skip if elevator is structurally useless for person j
                 (h_costs[j][elevator_node] == INF).
               – skip if weight would exceed capacity.
        """
        n_e = self.n_e
        n_p = self.n_p
        ELEV_OFFSET  = self.ELEV_OFFSET
        e_floors     = state[:n_e]
        p_locs       = state[n_e:]
        h_costs      = self.h_costs
        p_weights    = self.p_weights
        p_goals      = self.p_goals
        e_capacity   = self.e_capacity
        elev_useful  = self.elev_useful
        useful_tgts  = self.useful_targets
        useful_sets  = self.useful_sets
        move_str     = self.move_str
        enter_str    = self.enter_str
        exit_str     = self.exit_str

        # Current weight per elevator + whether it has any passenger.
        e_weights  = [0] * n_e
        e_occupied = [False] * n_e
        for j in range(n_p):
            loc = p_locs[j]
            if loc >= ELEV_OFFSET:
                ei = loc - ELEV_OFFSET
                e_weights[ei]  += p_weights[j]
                e_occupied[ei]  = True

        # Floors with at least one waiting (not in elevator, not done) person.
        waiting_floors = set()
        for j in range(n_p):
            loc = p_locs[j]
            if loc < ELEV_OFFSET and loc != p_goals[j]:
                waiting_floors.add(loc)

        successors = []

        # ── MOVE ────────────────────────────────────────────────────────
        for i in range(n_e):
            # Empty elevator: skip if no waiting person at any useful target.
            if not e_occupied[i] and not (useful_sets[i] & waiting_floors):
                continue
            cur = e_floors[i]
            mrow = move_str[i]
            for f in useful_tgts[i]:
                if f != cur:
                    successors.append((mrow[f], state[:i] + (f,) + state[i + 1:]))

        # ── EXIT ─────────────────────────────────────────────────────────
        for j in range(n_p):
            loc = p_locs[j]
            if loc < ELEV_OFFSET:
                continue
            ei = loc - ELEV_OFFSET
            f  = e_floors[ei]
            if h_costs[j][f] == INF:   # unreachable exit floor
                continue
            idx = n_e + j
            successors.append((exit_str[j][ei], state[:idx] + (f,) + state[idx + 1:]))

        # ── ENTER ────────────────────────────────────────────────────────
        for j in range(n_p):
            loc = p_locs[j]
            if loc >= ELEV_OFFSET:
                continue   # already in elevator
            if loc == p_goals[j]:
                continue   # at goal — entering would only add steps
            pw  = p_weights[j]
            eur = elev_useful[j]
            for i in range(n_e):
                if e_floors[i] != loc:
                    continue
                if not eur[i]:
                    continue   # elevator structurally can't help person j
                if e_weights[i] + pw > e_capacity[i]:
                    continue   # overweight
                idx = n_e + j
                successors.append((enter_str[j][i],
                                   state[:idx] + (ELEV_OFFSET + i,) + state[idx + 1:]))

        return successors

    # ── goal_test ─────────────────────────────────────────────────────────────
    def goal_test(self, state):
        # All person locations must equal their goal floors (< ELEV_OFFSET),
        # so this also implicitly checks that no one is still inside an elevator.
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
            F is *covered* iff at least one of:
              (a) source-cover: an elevator currently at F lies on a BFS
                  shortest path for some person waiting at F
                  [h_costs[j][elev_node] == h_costs[j][F] − 1]
              (b) goal-cover: a person whose goal is F is inside an
                  elevator that is currently at F.
            If F is uncovered, at least 1 future MOVE must bring some
            elevator to F. One MOVE visits exactly one floor.
            → strict_move_lb ≤ actual remaining MOVE count.

        The two terms bound disjoint action types (ENTER+EXIT vs MOVE),
        so their sum is a valid lower bound on the total remaining cost.
        Consistency (h(s) ≤ 1 + h(s') for every action) can be verified
        per action type: each action changes h by at most 1.
        """
        n_e      = self.n_e
        n_p      = self.n_p
        ELEV_OFFSET = self.ELEV_OFFSET
        height_p1   = self.height + 1
        state    = node.state
        e_floors = state[:n_e]
        p_locs   = state[n_e:]
        h_costs  = self.h_costs
        p_goals  = self.p_goals

        h_val = 0
        # Floor-indexed arrays avoid dict overhead in the inner loop.
        src_req  = [None] * height_p1   # src_req[f]  = [j, …] waiting at f
        goal_req = [None] * height_p1   # goal_req[f] = [j, …] with goal f
        any_unfinished = False

        for j in range(n_p):
            loc = p_locs[j]
            h_val += h_costs[j][loc]
            g = p_goals[j]
            if loc != g:
                any_unfinished = True
                if loc < ELEV_OFFSET:          # person waiting on floor
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

        # Group elevator indices by current floor (floor-indexed list).
        elevs_at = [None] * height_p1
        for i in range(n_e):
            f = e_floors[i]
            if 0 <= f < height_p1:
                if elevs_at[f] is None:
                    elevs_at[f] = [i]
                else:
                    elevs_at[f].append(i)

        uncovered = 0
        for f in range(height_p1):
            srcs = src_req[f]
            gs   = goal_req[f]
            if srcs is None and gs is None:
                continue

            # A floor is covered only when BOTH its requirements are met:
            #   src_covered: no waiting person, or some elevator at f is on a
            #                BFS shortest path for a waiting person (they can
            #                enter without a future MOVE).
            #   goal_covered: no delivery target, or the target person is
            #                 already inside an elevator sitting at f (EXIT
            #                 suffices, no future MOVE needed).
            # Using OR (old) lets EXIT reduce strict_move_lb and transfer_lb
            # simultaneously, violating consistency.  AND avoids that.
            src_covered  = srcs is None
            goal_covered = gs is None

            if not src_covered:
                e_list = elevs_at[f]
                if e_list is not None:
                    for i in e_list:
                        e_node = ELEV_OFFSET + i
                        for j in srcs:
                            if h_costs[j][e_node] == h_costs[j][f] - 1:
                                src_covered = True
                                break
                        if src_covered:
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
