# Event-Triggered Rescheduling Simulation

`event_triggered_rescheduling.py` simulates **event-driven rescheduling in collaborative logistics**. It demonstrates how delivery schedules can adapt to real-world disruptions such as vehicle breakdowns and traffic delays.

## Overview

This script:

- Creates depots, vehicles, and customers in a 2D area.
- Generates an initial schedule for each vehicle using a **nearest-neighbour greedy heuristic**.
- Simulates vehicle travel and triggers two events during the simulation:
  1. **Vehicle breakdown** – remaining deliveries are reassigned.
  2. **Traffic delay** – increases travel time on a specific road segment.
- Shows two rescheduling strategies:
  - **Local replanning:** only vehicles from the same depot can take over remaining deliveries.
  - **Collaborative handover:** vehicles from any depot can help.
- Outputs summary metrics and optional plots showing routes before and after events.

## Requirements

```bash
pip install numpy matplotlib
