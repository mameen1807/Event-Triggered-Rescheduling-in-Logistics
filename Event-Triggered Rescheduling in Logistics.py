"""
event_triggered_rescheduling.py

Event-triggered rescheduling simulation for collaborative logistics.

Run:
    python event_triggered_rescheduling.py

Dependencies:
    pip install numpy matplotlib

What it does (short):
 - Creates depots, vehicles and customers
 - Builds a simple greedy initial schedule for each vehicle (nearest-neighbour)
 - Simulates travel and at a given time injects 2 events:
     1) Vehicle breakdown (remaining deliveries must be reassigned)
     2) Traffic delay on a road segment (increases travel times on that edge)
 - Demonstrates two rescheduling modes:
     A) Local replanning (same-depot vehicles only)
     B) Collaborative handover (vehicles from any depot can help)
 - Outputs plots (before/after) and simple metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from math import hypot

random.seed(1)
np.random.seed(1)


# ---------------- Scenario Setup ----------------
NUM_DEPOTS = 2
VEHICLES_PER_DEPOT = 2
NUM_CUSTOMERS = 18
AREA = (0, 100)  # square area 0..100

# Create depot locations
depots = [(20, 50), (80, 50)]

# Create vehicles: each vehicle has id, depot, position, route (list of customer indices), status
vehicles = []
vid = 0
for d_idx, depot in enumerate(depots):
    for v in range(VEHICLES_PER_DEPOT):
        vehicles.append({
            "id": f"V{vid}",
            "depot": d_idx,
            "pos": depot,
            "route": [],        # will be assigned
            "completed": [],    # delivered customer indices
            "broken": False
        })
        vid += 1

# Create customer locations and earliest creation time
customers = []
for i in range(NUM_CUSTOMERS):
    # cluster customers nearer to each depot alternately to simulate realistic distribution
    if i % 2 == 0:
        center = depots[0]
        spread = 18
    else:
        center = depots[1]
        spread = 18
    x = center[0] + random.uniform(-spread, spread)
    y = center[1] + random.uniform(-spread, spread)
    customers.append({"id": f"C{i}", "pos": (x, y), "assigned": None, "delivered": False})

# Metric helpers
def dist(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

# ---------------- Initial Assignment (balanced greedy) ----------------
# Assign customers to nearest depot, then distribute to its vehicles evenly
customers_by_depot = {d: [] for d in range(NUM_DEPOTS)}
for idx, cust in enumerate(customers):
    dists = [dist(cust["pos"], depots[d]) for d in range(NUM_DEPOTS)]
    nearest_d = int(np.argmin(dists))
    customers_by_depot[nearest_d].append(idx)

# Distribute customers to vehicles per depot using simple round-robin then perform a nearest-neighbour route
for d_idx in range(NUM_DEPOTS):
    depot_vehicle_ids = [v for v in vehicles if v["depot"] == d_idx]
    assigned = customers_by_depot[d_idx]
    # round-robin assignment to vehicles to start balanced
    for i, cidx in enumerate(assigned):
        target_v = depot_vehicle_ids[i % len(depot_vehicle_ids)]
        target_v["route"].append(cidx)
        customers[cidx]["assigned"] = target_v["id"]

# Improve each vehicle route via a simple nearest-neighbour ordering from depot
def nn_route(depot_pos, cust_indices):
    if not cust_indices:
        return []
    route = []
    remaining = cust_indices.copy()
    cur = depot_pos
    while remaining:
        dlist = [(dist(cur, customers[i]["pos"]), i) for i in remaining]
        dlist.sort()
        _, choose = dlist[0]
        route.append(choose)
        remaining.remove(choose)
        cur = customers[choose]["pos"]
    return route

for v in vehicles:
    v["route"] = nn_route(depots[v["depot"]], v["route"])

# Save a snapshot of the initial assignment for plotting later
import copy
vehicles_initial = copy.deepcopy(vehicles)
customers_initial = copy.deepcopy(customers)

# ---------------- Travel Time / Simulation Helpers ----------------
SPEED = 40.0  # units per hour (arbitrary)
def travel_time(a, b, speed=SPEED, multiplier=1.0):
    return dist(a, b) / speed * multiplier  # hours

# Track simulated clock (hours)
SIM_TIME_HORIZON = 6.0
TIME_STEP = 0.1  # hours per simulation step

# Event definitions
EVENTS = [
    {"time": 1.5, "type": "breakdown", "vehicle_id": vehicles[1]["id"]},  # break vehicle V1 at t=1.5
    {"time": 2.6, "type": "traffic", "edge": ( (40,50), (60,50) ), "multiplier": 3.0}  # increase travel time across a corridor
]

# For simplicity we model "traffic" as increasing travel time multiplier when a route segment crosses the given bounding box.
def traffic_multiplier(a, b, traffic_event):
    # if segment crosses x between 40..60 and y between 45..55 then multiplier else 1
    (x1,y1),(x2,y2)=(a,b)
    if (min(x1,x2) <= 60 and max(x1,x2) >= 40) and (min(y1,y2) <= 55 and max(y1,y2) >= 45):
        return traffic_event["multiplier"]
    return 1.0

# ---------------- Rescheduling functions ----------------
def reassign_remaining_orders(broken_vehicle_id, vehicles_list, customers_list, allow_collaboration=False):
    """
    When a vehicle breaks, reassign its remaining (not yet delivered) customers.
    If allow_collaboration = False -> only same-depot vehicles will accept.
    If True -> any vehicle (from other depots too) may accept.
    Strategy: insert each broken customer's location into the route of the vehicle
    that yields the minimal incremental travel distance (simple insertion heuristic).
    """
    # find broken vehicle object
    bv = next((v for v in vehicles_list if v["id"]==broken_vehicle_id), None)
    if bv is None: 
        return 0
    remaining = [c for c in bv["route"] if not customers_list[c]["delivered"]]
    if not remaining:
        return 0
    reassign_count = 0
    for cust_idx in remaining:
        best_increase = float("inf")
        best_vehicle = None
        best_pos = None
        for v in vehicles_list:
            if v["id"] == broken_vehicle_id:
                continue
            if (not allow_collaboration) and (v["depot"] != bv["depot"]):
                continue
            # consider inserting cust_idx at every possible position in v.route
            route = v["route"]
            base_cost = 0.0
            # compute current route travel distance (depot -> ... -> depot)
            cur = depots[v["depot"]]
            for r in route:
                base_cost += dist(cur, customers[r]["pos"])
                cur = customers[r]["pos"]
            base_cost += dist(cur, depots[v["depot"]])
            # try inserts
            for pos in range(len(route)+1):
                new_route = route[:pos] + [cust_idx] + route[pos:]
                new_cost = 0.0
                cur = depots[v["depot"]]
                for r in new_route:
                    new_cost += dist(cur, customers[r]["pos"])
                    cur = customers[r]["pos"]
                new_cost += dist(cur, depots[v["depot"]])
                incr = new_cost - base_cost
                if incr < best_increase:
                    best_increase = incr
                    best_vehicle = v
                    best_pos = pos
        # assign to best_vehicle
        if best_vehicle:
            best_vehicle["route"].insert(best_pos, cust_idx)
            customers_list[cust_idx]["assigned"] = best_vehicle["id"]
            reassign_count += 1
    # clear remaining from broken vehicle's route
    bv["route"] = [r for r in bv["route"] if customers_list[r]["delivered"]]
    return reassign_count

# ---------------- Simulation run (executes timeline, triggers events, uses rescheduling) ----------------
def run_simulation(vehicles_state, customers_state, events, collaborative=False, verbose=True):
    # deep copy passed in (we modify)
    import copy
    v_state = copy.deepcopy(vehicles_state)
    c_state = copy.deepcopy(customers_state)
    t = 0.0
    event_idx = 0
    total_distance = 0.0
    reassignments = 0
    completion_times = {i: None for i in range(len(c_state))}
    # helper to compute route distance for metrics
    def route_distance_for_vehicle(v):
        cur = depots[v["depot"]]
        s = 0.0
        for r in v["route"]:
            s += dist(cur, customers[r]["pos"])
            cur = customers[r]["pos"]
        s += dist(cur, depots[v["depot"]])
        return s

    # simulate until all customers delivered or horizon exceeded
    # We model delivery completion by assuming each vehicle completes its whole route sequentially (not continuous time stepping for simplicity)
    # But we will trigger events at given times and replan mid-way.
    # For each vehicle, we compute time to finish step-by-step, compare with event times, and if event occurs while it's en route, we trigger replanning.
    # This is a simplified but explainable model.
    current_time = 0.0
    # Build a per-vehicle timeline of (segment travel time, customer idx) and consume until events handled
    # We'll iterate vehicles in round-robin and advance until all customers delivered or events processed and timeline ended.
    # Simpler approach: process in chronological order next segment across all vehicles.
    # Build initial next-segment list
    # For each vehicle, pointer to next customer index in route
    pointers = {v["id"]: 0 for v in v_state}
    # positions of vehicles start at depot
    positions = {v["id"]: depots[v["depot"]] for v in v_state}
    # last event times to control single-trigger
    events_handled = [False]*len(events)

    while True:
        # check termination: all delivered
        if all(c["delivered"] for c in c_state):
            break
        # if no more segments (all routes empty), break
        active_segments = []
        for v in v_state:
            if v["broken"]:
                continue
            pid = pointers[v["id"]]
            if pid < len(v["route"]):
                cust_i = v["route"][pid]
                a = positions[v["id"]]
                b = customers[cust_i]["pos"]
                # compute travel time multiplier from traffic events if event already active
                mul = 1.0
                # check any traffic event that has started and not ended (we assume permanent after start)
                for ev in events:
                    if ev["type"]=="traffic" and current_time >= ev["time"]:
                        mul = mul * ev.get("multiplier",1.0)
                seg_time = travel_time(a, b, SPEED, multiplier=mul)
                active_segments.append( (current_time + seg_time, v, cust_i, seg_time, a, b) )
        if not active_segments:
            break
        # pick soonest finishing segment
        active_segments.sort(key=lambda x: x[0])
        finish_time, v_obj, cust_i, seg_time, a, b = active_segments[0]
        # advance current_time to finish_time
        # But first, if there are events between current_time and finish_time, handle them in chronological order
        # Handle events whose time <= finish_time and not yet handled:
        for ei, ev in enumerate(events):
            if (not events_handled[ei]) and (ev["time"] <= finish_time):
                # advance time to ev time and handle
                current_time = ev["time"]
                if ev["type"] == "breakdown":
                    # find vehicle and mark broken
                    for vv in v_state:
                        if vv["id"] == ev["vehicle_id"]:
                            vv["broken"] = True
                            if verbose: print(f"[t={current_time:.2f}h] Event: Breakdown of {vv['id']}.")
                            # reassign remaining jobs from vv
                            re = reassign_remaining_orders(vv["id"], v_state, c_state, allow_collaboration=collaborative)
                            reassignments += re
                            if verbose: print(f"  Reassigned {re} remaining orders (collab={collaborative}).")
                            break
                elif ev["type"] == "traffic":
                    if verbose: print(f"[t={current_time:.2f}h] Event: Traffic slowdown in corridor (multiplier={ev.get('multiplier')}).")
                    # traffic is handled implicitly in travel_time via multiplier
                events_handled[ei] = True
        # after handling all events up to finish_time, now complete the selected segment
        current_time = finish_time
        # deliver cust_i by vehicle v_obj if not already delivered and vehicle not broken
        if v_obj["broken"]:
            # segment was interrupted by breakdown, skip delivering
            continue
        if not c_state[cust_i]["delivered"]:
            c_state[cust_i]["delivered"] = True
            completion_times[cust_i] = current_time
            v_obj["completed"].append(cust_i)
            # advance pointer
            pointers[v_obj["id"]] += 1
            # vehicle position moves to customer pos
            positions[v_obj["id"]] = customers[cust_i]["pos"]
            total_distance += dist(a,b)
        else:
            # already delivered (maybe reassigned & delivered by another), just advance pointer
            pointers[v_obj["id"]] += 1
            positions[v_obj["id"]] = customers[cust_i]["pos"]
            total_distance += dist(a,b)

    # compute metrics
    delivered_count = sum(1 for c in c_state if c["delivered"])
    avg_completion = np.mean([t for t in completion_times.values() if t is not None]) if delivered_count>0 else None

    metrics = {
        "delivered": delivered_count,
        "total_distance": total_distance,
        "reassignments": reassignments,
        "avg_completion_time_h": avg_completion
    }
    return metrics, v_state, c_state

# ---------------- Run two scenarios: local replanning vs collaborative handover ----------------
print("Initial assignment summary:")
for v in vehicles_initial:
    print(f"  {v['id']} depot={v['depot']} route={[customers[r]['id'] for r in v['route']]}")

print("\n--- Running LOCAL replanning (same-depot only) ---")
metrics_local, vehicles_local, customers_local = run_simulation(vehicles_initial, customers_initial, EVENTS, collaborative=False, verbose=True)
print("Local metrics:", metrics_local)

print("\n--- Running COLLABORATIVE handover (any depot may help) ---")
# need fresh copies: restore initial state
vehicles_for_collab = copy.deepcopy(vehicles_initial)
customers_for_collab = copy.deepcopy(customers_initial)
metrics_collab, vehicles_collab, customers_collab = run_simulation(vehicles_for_collab, customers_for_collab, EVENTS, collaborative=True, verbose=True)
print("Collaborative metrics:", metrics_collab)

# ---------------- Plotting helper ----------------
def plot_routes(vehicles_state, customers_state, title):
    plt.figure(figsize=(8,6))
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    # plot depots
    for i,d in enumerate(depots):
        plt.scatter(d[0], d[1], marker='s', s=120, color='k')
        plt.text(d[0]+1, d[1]+1, f"Depot {i}", fontsize=10)
    # customers
    for ci, cust in enumerate(customers_state):
        col = 'gray' if cust["delivered"] else 'tab:gray'
        plt.scatter(cust["pos"][0], cust["pos"][1], marker='o', color=col, s=40)
        plt.text(cust["pos"][0]+0.8, cust["pos"][1]+0.8, cust["id"], fontsize=8)
    # vehicle routes
    for idx, v in enumerate(vehicles_state):
        route_coords = [depots[v["depot"]]] + [customers_state[c]["pos"] for c in v["route"]] + [depots[v["depot"]]]
        xs = [p[0] for p in route_coords]
        ys = [p[1] for p in route_coords]
        plt.plot(xs, ys, '-o', color=colors[idx % len(colors)], label=v["id"])
        plt.text(xs[0]+0.5, ys[0]+0.5, v["id"], fontsize=9)
        if v.get("broken"):
            plt.text(xs[0]-2, ys[0]-2, "(BROKEN)", color='red')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_routes(vehicles_initial, customers_initial, "Initial planned routes (before events)")
plot_routes(vehicles_local, customers_local, "After LOCAL replanning")
plot_routes(vehicles_collab, customers_collab, "After COLLABORATIVE replanning")

# ---------------- Final printed comparison ----------------
print("\nComparison summary:")
print("Scenario\tDelivered\tTotalDist\tReassignments\tAvgCompletion_h")
print(f"Local\t\t{metrics_local['delivered']}\t\t{metrics_local['total_distance']:.1f}\t\t{metrics_local['reassignments']}\t\t{metrics_local['avg_completion_time_h']:.2f}")
print(f"Collaborative\t{metrics_collab['delivered']}\t\t{metrics_collab['total_distance']:.1f}\t\t{metrics_collab['reassignments']}\t\t{metrics_collab['avg_completion_time_h']:.2f}")