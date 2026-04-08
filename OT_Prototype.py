import os
import sys
import traci
import sumolib
import networkx as nx
import numpy as np
import ot
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SUMO_BINARY = r"D:\WORK\SUMO\bin\sumo-gui.exe"
SUMO_CONFIG = r"D:\PROJECTS\CCP PJCTS\SEM 4\LynxSpiderWeb\Intersection\Intersection.sumocfg"
NETWORK_FILE = r"D:\PROJECTS\CCP PJCTS\SEM 4\LynxSpiderWeb\Intersection\Intersection.net.xml"

class TraCISimulator:
    """Module 1: Traffic Simulation Layer"""
    def __init__(self, sumo_binary, sumo_config):
        self.sumo_cmd = [sumo_binary, "-c", sumo_config]
        self.tls_ids = []
        self.tls_lanes = {}

    def start(self):
        traci.start(self.sumo_cmd)
        self.tls_ids = traci.trafficlight.getIDList()
        # Map controlled lanes to each TLS
        for tls in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls)
            self.tls_lanes[tls] = list(set(lanes))
        print(f"Simulation started. Controlled TLS: {self.tls_ids}")

    def step(self):
        traci.simulationStep()

    def get_traffic_data(self):
        """Collect vehicle counts and halting vehicles per intersection."""
        data = {}
        for tls in self.tls_ids:
            count = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.tls_lanes[tls])
            halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.tls_lanes[tls])
            data[tls] = {"count": count, "halting": halting}
        return data

    def close(self):
        traci.close()

class TrafficGraph:
    """Module 2: Graph Representation"""
    def __init__(self, net_file):
        self.net = sumolib.net.readNet(net_file)
        self.G = nx.DiGraph()
        self.junction_ids = []
        self.pos = {}
        self._build_graph()

    def _build_graph(self):
        # Extract junctions that are traffic lights
        for junction in self.net.getNodes():
            if junction.getType() == "traffic_light":
                jid = junction.getID()
                self.junction_ids.append(jid)
                coord = junction.getCoord()
                self.pos[jid] = coord
                self.G.add_node(jid, pos=coord)

        # Build edges based on road connections
        for u in self.junction_ids:
            for v in self.junction_ids:
                if u == v: continue
                # Basic Euclidean for connectivity if complex sumolib paths are overkill
                # But let's try to find if there's a road between them
                u_node = self.net.getNode(u)
                v_node = self.net.getNode(v)
                # Check for edges between them
                for edge in u_node.getOutgoing():
                    if edge.getToNode().getID() == v:
                        self.G.add_edge(u, v, weight=edge.getLength())

    def get_cost_matrix(self, junction_order):
        """Build normalized cost matrix based on shortest paths."""
        n = len(junction_order)
        C = np.zeros((n, n))
        
        # Calculate all-pairs shortest paths
        all_paths = dict(nx.all_pairs_dijkstra_path_length(self.G))
        
        for i, u in enumerate(junction_order):
            for j, v in enumerate(junction_order):
                if u == v:
                    C[i, j] = 0
                elif v in all_paths.get(u, {}):
                    C[i, j] = all_paths[u][v]
                else:
                    # Fallback to Euclidean if no path (unlikely in small connected network)
                    pos_u = self.pos[u]
                    pos_v = self.pos[v]
                    dist = np.sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2)
                    C[i, j] = dist * 2 # Penalty for no direct path
        
        # Normalize
        if C.max() > 0:
            C = C / C.max()
        return C

class DistributionModeler:
    """Module 3: Traffic Distribution Modeling"""
    def __init__(self, junction_ids):
        self.junction_ids = junction_ids

    def get_distribution(self, traffic_data):
        counts = np.array([traffic_data[jid]["count"] for jid in self.junction_ids], dtype=float)
        total = counts.sum()
        
        if total == 0:
            return None, 0
        
        # μ = vehicle count / total vehicles
        mu = counts / total
        return mu, total

class OTEngine:
    """Module 4: Optimal Transport Engine"""
    def __init__(self, cost_matrix, reg=0.1):
        self.C = cost_matrix
        self.reg = reg

    def compute_transport(self, mu, target):
        # Sinkhorn algorithm
        gamma = ot.sinkhorn(mu, target, self.C, self.reg)
        wasserstein = np.sum(gamma * self.C)
        return gamma, wasserstein

    def generate_target(self, mu, strategy="balance", epsilon=0.1):
        if strategy == "balance":
            # Desired is uniform distribution (target = [1/n, ..., 1/n])
            desired = np.ones_like(mu) / len(mu)
        else:
            desired = mu # Default to no change
            
        # Blended update rule: target = (1 - epsilon) * mu + epsilon * desired
        target = (1 - epsilon) * mu + epsilon * desired
        # Re-normalize just in case of precision issues
        target = target / target.sum()
        return target

class SignalController:
    """Module 5: Signal Control Layer"""
    def __init__(self, tls_ids, alpha=5.0, smoothing=0.8):
        self.tls_ids = tls_ids
        self.alpha = alpha  # Proportional gain
        self.smoothing = smoothing
        self.last_durations = {tid: 42.0 for tid in tls_ids} # Start with default 42
        self.min_green = 10
        self.max_green = 60

    def adjust_signals(self, mu, target, traffic_data):
        for i, tid in enumerate(self.tls_ids):
            # Proportional control: green_adjustment = alpha * (mu[i] - target[i])
            # Higher current mu relative to target means we need MORE green time
            diff = mu[i] - target[i]
            
            # Safety checks: locking prevention (high queue/halting number)
            halting = traffic_data[tid]["halting"]
            if halting > 10:
                # Add extra compensation for high congestion
                diff += 0.05 

            # Computed adjustment
            adjustment = self.alpha * diff * 100 # Scaling for probability diff
            new_dur = self.last_durations[tid] + adjustment
            
            # Clamp
            new_dur = max(self.min_green, min(self.max_green, new_dur))
            
            # Smoothing (EMA)
            applied_dur = self.smoothing * self.last_durations[tid] + (1 - self.smoothing) * new_dur
            
            # Apply to green phases only
            state = traci.trafficlight.getRedYellowGreenState(tid)
            if 'y' not in state.lower() and 'r' not in state.lower():
                 # Simple heuristic: if the state is mostly green, adjust it
                 # In a static program, we can just set the next phase duration
                 traci.trafficlight.setPhaseDuration(tid, int(applied_dur))
            
            self.last_durations[tid] = applied_dur

def run_prototype():
    # 1. Init Simulation
    sim = TraCISimulator(SUMO_BINARY, SUMO_CONFIG)
    sim.start()
    
    # 2. Init Graph & Cost Matrix
    graph = TrafficGraph(NETWORK_FILE)
    C = graph.get_cost_matrix(sim.tls_ids)
    
    # 3. Init Engine & Logic
    modeler = DistributionModeler(sim.tls_ids)
    engine = OTEngine(C)
    controller = SignalController(sim.tls_ids)
    
    # Monitoring lists
    wasserstein_history = []
    
    try:
        for step in range(5000):
            sim.step()
            
            if step % 5 == 0: # Optimization step interval
                # Collect and Model Distributions
                traffic_data = sim.get_traffic_data()
                mu, total_traffic = modeler.get_distribution(traffic_data)
                
                if mu is not None and total_traffic > 5:
                    # Compute OT
                    target = engine.generate_target(mu, strategy="balance", epsilon=0.2)
                    gamma, W = engine.compute_transport(mu, target)
                    wasserstein_history.append(W)
                    
                    # Control Signals
                    controller.adjust_signals(mu, target, traffic_data)
                    
                    if step % 50 == 0:
                        print(f"Step {step} | Total Vehicles: {total_traffic} | Wasserstein Dist: {W:.4f}")
                        # print(f"  Distribution: {['%.2f' % x for x in mu]}")

    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        sim.close()
        print("Simulation concluded.")
        
        if wasserstein_history:
            plt.figure(figsize=(10, 5))
            plt.plot(wasserstein_history)
            plt.title("Wasserstein Distance (Imbalance) Over Time")
            plt.xlabel("Optimization Steps")
            plt.ylabel("W Distance")
            plt.grid(True)
            plt.savefig("congestion_evolution.png")
            print("Evolution plot saved as congestion_evolution.png")

if __name__ == "__main__":
    run_prototype()
