# fuzzy_allocator_server.py
from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# --- Fuzzy System Definition (runs once at startup) ---
priority = ctrl.Antecedent(np.arange(0, 11, 1), 'priority')
bandwidth = ctrl.Consequent(np.arange(0, 101, 1), 'bandwidth')
cpu = ctrl.Consequent(np.arange(0, 101, 1), 'cpu')
priority['low'] = fuzz.trimf(priority.universe, [0, 0, 4])
priority['medium'] = fuzz.trimf(priority.universe, [2, 5, 8])
priority['high'] = fuzz.trimf(priority.universe, [6, 10, 10])
for var in [bandwidth, cpu]:
    var['low'] = fuzz.trimf(var.universe, [0, 0, 40])
    var['medium'] = fuzz.trimf(var.universe, [30, 50, 70])
    var['high'] = fuzz.trimf(var.universe, [60, 100, 100])
rule1 = ctrl.Rule(priority['low'], [bandwidth['low'], cpu['low']])
rule2 = ctrl.Rule(priority['medium'], [bandwidth['medium'], cpu['medium']])
rule3 = ctrl.Rule(priority['high'], [bandwidth['high'], cpu['high']])
allocation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
print("--- Fuzzy Allocator Server ---")
print("Fuzzy control system ready.")

# --- API Endpoint ---
@app.route('/allocate_resources', methods=['POST'])
def allocate_resources():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    priority_label = str(data.get("priority", "LOW")).upper()
    priority_map = {"LOW": 2, "MEDIUM": 5, "HIGH": 8}
    if priority_label not in priority_map:
        return jsonify({"error": "priority must be LOW, MEDIUM, or HIGH"}), 400

    allocation_sim = ctrl.ControlSystemSimulation(allocation_ctrl)
    allocation_sim.input['priority'] = priority_map[priority_label]
    allocation_sim.compute()
    bw = round(allocation_sim.output['bandwidth'], 2)
    cpu_alloc = round(allocation_sim.output['cpu'], 2)
    decision = {
        "priority": priority_label,
        "bandwidth_allocation": f"{bw}%",
        "cpu_allocation": f"{cpu_alloc}%",
        "decision": f"Allocate {bw}% bandwidth and {cpu_alloc}% CPU for {priority_label} priority"
    }
    return jsonify(decision)

# --- Run Server ---

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)