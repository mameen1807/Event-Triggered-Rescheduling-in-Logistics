# Event-Triggered Rescheduling Simulation

This project demonstrates a **collaborative logistics simulation** with dynamic events.  
It models depots, vehicles, and customers, and shows how local and collaborative rescheduling can respond to unexpected disruptions.

---

## 📌 Overview
- Simulates **delivery operations** for multiple depots with assigned vehicles.
- Injects **events mid-simulation**, such as:
  1. **Vehicle breakdowns** — remaining deliveries must be reassigned.
  2. **Traffic delays** — increases travel time along specific corridors.
- Demonstrates two rescheduling strategies:
  - **Local replanning**: only vehicles from the same depot can take over.
  - **Collaborative handover**: any vehicle, from any depot, can assist.
- Calculates metrics like total distance traveled, reassignments, and average completion time.
- Provides **visualizations of routes** before and after events.

---

## 🎯 Objectives
- Understand the impact of **unplanned disruptions** on delivery schedules.
- Compare **local vs. collaborative rescheduling** strategies.
- Visualize **dynamic route changes** and operational metrics.

---

## 🛠️ Methodology
1. Define depots, vehicles, and customer locations.
2. Assign initial routes using a **nearest-neighbour heuristic**.
3. Simulate travel, injecting events at specific times.
4. Reassign remaining deliveries using:
   - Local replanning
   - Collaborative handover
5. Track metrics and plot routes for comparison.

---

## 📊 Key Metrics
- **Delivered**: total customers successfully served.
- **Total distance**: cumulative travel distance of all vehicles.
- **Reassignments**: number of orders reassigned due to events.
- **Average completion time**: mean delivery time for all served customers.

---

## 📷 Example Output
- Initial planned routes
- Routes after **LOCAL replanning**
- Routes after **COLLABORATIVE replanning**

---

## 🚀 How to Run
```bash
python event_triggered_rescheduling.py
