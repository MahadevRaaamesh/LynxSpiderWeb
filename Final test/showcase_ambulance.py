import os
import sys
import traci
import sumolib
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import ot
from collections import deque

# --- CONFIGURATION ---
SUMO_BINARY = r"D:\WORK\SUMO\bin\sumo.exe"
SUMO_CONFIG = r"d:\PROJECTS\CCP PJCTS\SEM 4\LynxSpiderWeb\Intersection\Intersection.sumocfg"
NETWORK_FILE = r"d:\PROJECTS\CCP PJCTS\SEM 4\LynxSpiderWeb\Intersection\Intersection.net.xml"
MODEL_LOAD_PATH = "ot_hybrid_dqn_weights.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LAYER 1: NEURAL NETWORK (DQN) ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

# --- LAYER 2: SIMULATION & UTILS ---
class TraCISimulator:
    def __init__(self, sumo_binary, sumo_config, network_file=None):
        self.sumo_cmd = [sumo_binary, "-c", sumo_config, "--waiting-time-memory", "1000"]
        self.tls_ids = []
        self.tls_lanes = {}
        self.edges = []
        self.net = None
        if network_file:
            self.net = sumolib.net.readNet(network_file)

    def start(self):
        traci.start(self.sumo_cmd)
        self.tls_ids = traci.trafficlight.getIDList()
        for tls in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls)
            self.tls_lanes[tls] = list(set(lanes))
            
        self.edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]

    def step(self):
        traci.simulationStep()

    def get_traffic_data(self):
        data = {}
        for tls in self.tls_ids:
            count = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.tls_lanes[tls])
            halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.tls_lanes[tls])
            data[tls] = {"count": count, "halting": halting}
        return data

    def spawn_ambulance(self, amb_id="amb_0"):
        if len(self.edges) < 2: return False
        
        try:
            if "ambulance_type" not in traci.vehicletype.getIDList():
                traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_type")
                traci.vehicletype.setVehicleClass("ambulance_type", "emergency")
                traci.vehicletype.setColor("ambulance_type", (255, 0, 0, 255))
                traci.vehicletype.setLength("ambulance_type", 6.5)
                traci.vehicletype.setMaxSpeed("ambulance_type", 30.0)
                traci.vehicletype.setShapeClass("ambulance_type", "emergency")
        except:
            pass
            
        for _ in range(20):
            src = random.choice(self.edges)
            dst = random.choice(self.edges)
            if src == dst: continue
            
            try:
                route = traci.simulation.findRoute(src, dst)
                if route and route.edges and len(route.edges) > 1:
                    route_id = f"route_{amb_id}_{random.randint(1000, 9999)}"
                    traci.route.add(route_id, route.edges)
                    traci.vehicle.add(amb_id, route_id, typeID="ambulance_type")
                    traci.vehicle.setSpeedFactor(amb_id, 1.5)
                    traci.vehicle.setLaneChangeMode(amb_id, 0b100000101001)
                    print(f"** AMBULANCE SPAWNED: {amb_id} traveling from {src} to {dst} **")
                    return True
            except:
                pass
        return False

    def get_ambulance_lanes(self):
        lanes = []
        try:
            for vid in traci.vehicle.getIDList():
                vtype = traci.vehicle.getTypeID(vid)
                if vtype == "ambulance_type" or "amb" in vid:
                    l = traci.vehicle.getLaneID(vid)
                    if l and not l.startswith(":"):
                        lanes.append((vid, l))
        except:
            pass
        return lanes

    def get_upcoming_corridor(self, amb_id, look_ahead=2):
        """Returns the next look_ahead (tls_id, distance) pairs for the ambulance."""
        corridor = []
        if amb_id not in traci.vehicle.getIDList(): return []
        try:
            route = traci.vehicle.getRoute(amb_id)
            curr_edge = traci.vehicle.getRoadID(amb_id)
            if curr_edge.startswith(":"): return []
            
            try:
                idx = route.index(curr_edge)
                upcoming_edges = route[idx:]
            except ValueError:
                return []

            dist_acc = 0
            found_tls = 0
            
            # Start from current position on current edge
            pos_on_edge = traci.vehicle.getLanePosition(amb_id)
            edge_obj = self.net.getEdge(curr_edge)
            dist_acc = edge_obj.getLength() - pos_on_edge
            
            for next_edge_id in upcoming_edges:
                e = self.net.getEdge(next_edge_id)
                next_node = e.getToNode()
                
                if next_node.getType() == "traffic_light":
                    tls_id = next_node.getID()
                    if tls_id in self.tls_ids:
                        corridor.append((tls_id, dist_acc))
                        found_tls += 1
                        if found_tls >= look_ahead:
                            break
                
                if next_edge_id != curr_edge: # Only add full lengths of subsequent edges
                    dist_acc += e.getLength()
                    if dist_acc > 500: break # Don't look too far ahead
        except:
            pass
        return corridor

class TrafficGraph:
    def __init__(self, net_file):
        self.net = sumolib.net.readNet(net_file)
        self.G = nx.DiGraph()
        self.junction_ids = []
        self.pos = {}
        self._build_graph()

    def _build_graph(self):
        for junction in self.net.getNodes():
            if junction.getType() == "traffic_light":
                jid = junction.getID()
                self.junction_ids.append(jid)
                self.pos[jid] = junction.getCoord()
                self.G.add_node(jid)
        for u in self.junction_ids:
            u_node = self.net.getNode(u)
            for edge in u_node.getOutgoing():
                v = edge.getToNode().getID()
                if v in self.junction_ids:
                    self.G.add_edge(u, v, weight=edge.getLength())

    def get_cost_matrix(self, junction_order):
        n = len(junction_order)
        C = np.zeros((n, n))
        all_paths = dict(nx.all_pairs_dijkstra_path_length(self.G))
        for i, u in enumerate(junction_order):
            for j, v in enumerate(junction_order):
                if u == v: C[i, j] = 0
                elif v in all_paths.get(u, {}): C[i, j] = all_paths[u][v]
                else: 
                    p1, p2 = self.pos[u], self.pos[v]
                    C[i, j] = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 5
        if C.max() > 0: C /= C.max()
        return C

# --- LAYER 3: OPTIMAL TRANSPORT ---
class OTEngine:
    def __init__(self, cost_matrix, reg=0.5):
        self.C = cost_matrix
        self.reg = reg

    def compute_ot_guidance(self, mu, target):
        # Add epsilon for numerical stability
        mu = mu + 1e-9
        mu = mu / mu.sum()
        target = target + 1e-9
        target = target / target.sum()
        
        gamma = ot.sinkhorn(mu, target, self.C, self.reg)
        wd = np.sum(gamma * self.C)
        guidance = target - mu 
        return gamma, wd, guidance

# --- LAYER 4: HYBRID COORDINATOR ---
class HybridAgent:
    def __init__(self, tls_id, state_dim, action_dim=2):
        self.tls_id = tls_id
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer()
        self.epsilon = 0.05
        self.gamma = 0.95
        self.last_phase = -1
        self.time_in_phase = 0

    def select_action(self, state, train=False):
        if train and random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(DEVICE)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

class HybridCoordinator:
    def __init__(self, sim, graph, engine):
        self.sim = sim
        self.graph = graph
        self.engine = engine
        self.agents = {}
        self.tls_ids = []

    def _init_agents(self):
        self.tls_ids = self.sim.tls_ids
        for tid in self.tls_ids:
            if tid not in self.agents:
                n_lanes = len(self.sim.tls_lanes[tid])
                state_dim = n_lanes + 1 + 1 + 1
                self.agents[tid] = HybridAgent(tid, state_dim)

    def load_weights(self, path):
        if os.path.exists(path):
            try:
                weights = torch.load(path, map_location=DEVICE)
                for tid, agent in self.agents.items():
                    if tid in weights:
                        agent.policy_net.load_state_dict(weights[tid])
                        agent.target_net.load_state_dict(weights[tid])
                print(f"\n[OK] Weights loaded from {path}")
            except Exception as e:
                print(f"[ERROR] Error loading weights: {e}")
        else:
            print(f"[WARN] No weights found at {path}. Using random initialization.")

    def get_state(self, tid, guidance_val):
        lanes = self.sim.tls_lanes[tid]
        halting = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
        phase = traci.trafficlight.getPhase(tid)
        
        agent = self.agents[tid]
        if phase != agent.last_phase:
            agent.time_in_phase = 0
            agent.last_phase = phase
        else:
            agent.time_in_phase += 1
            
        state = halting + [phase, agent.time_in_phase, guidance_val]
        return np.array(state, dtype=np.float32)

    def run_epoch(self, steps=2000, train=False, spawn_amb_interval=300):
        self.sim.start()
        self._init_agents()
        self.load_weights(MODEL_LOAD_PATH)
        
        self.tls_ids = self.sim.tls_ids 
        C = self.graph.get_cost_matrix(self.tls_ids)
        self.engine.C = C 
        
        traffic_data = self.sim.get_traffic_data()
        mu = np.array([traffic_data[t]["count"] for t in self.tls_ids], dtype=float)
        mu = mu / (mu.sum() + 1e-9)
        target_dist = np.ones_like(mu) / (len(mu) + 1e-9)
        _, _, guidance = self.engine.compute_ot_guidance(mu, target_dist)
        
        states = {tid: self.get_state(tid, guidance[i]) for i, tid in enumerate(self.tls_ids)}

        amb_counter = 0
        for step in range(steps):
            if step > 0 and step % spawn_amb_interval == 0:
                if self.sim.spawn_ambulance(f"amb_{amb_counter}"):
                    amb_counter += 1

            amb_lanes_info = self.sim.get_ambulance_lanes()
            active_amb_edges = {traci.lane.getEdgeID(l) for _, l in amb_lanes_info}
            
            actions = {}
            corridor_overrides = {} # tid -> action (0/1)

            # --- AMBULANCE INTELLIGENCE: Corridor Tracking ---
            active_amb_edges = set()
            for vid, l in amb_lanes_info:
                try:
                    active_amb_edges.add(traci.lane.getEdgeID(l))
                    corridor = self.sim.get_upcoming_corridor(vid, look_ahead=2)
                    
                    if not corridor: continue
                    
                    for i, (tid, dist) in enumerate(corridor):
                        # Force action if close enough
                        if dist < 150:
                            print(f"[CORRIDOR] Clearance for {vid} at {tid} ({int(dist)}m)")
                            # Logic to determine if we should switch or hold green
                            # (Detailed preemption moved here inside decision phase)
                            links = traci.trafficlight.getControlledLinks(tid)
                            amb_path_active = False
                            for link_group in links:
                                if link_group and traci.lane.getEdgeID(link_group[0][0]) in active_amb_edges:
                                    amb_path_active = True
                                    break
                            
                            state_str = traci.trafficlight.getRedYellowGreenState(tid)
                            is_green = False
                            for group_idx, link_group in enumerate(links):
                                if link_group and traci.lane.getEdgeID(link_group[0][0]) in active_amb_edges:
                                    if group_idx < len(state_str) and state_str[group_idx].lower() == 'g':
                                        is_green = True
                                        break
                            
                            corridor_overrides[tid] = 0 if is_green else 1
                except:
                    pass

            for tid in self.tls_ids:
                if tid in corridor_overrides:
                    agent_action = corridor_overrides[tid]
                else:
                    agent_action = self.agents[tid].select_action(states[tid], train)
                
                actions[tid] = agent_action

            for tid in self.tls_ids:
                if actions[tid] == 1: 
                    try:
                        state_str = traci.trafficlight.getRedYellowGreenState(tid)
                        min_time = 3 if any(traci.lane.getEdgeID(lnk[0][0]) in active_amb_edges for lnk in traci.trafficlight.getControlledLinks(tid) if lnk) else 10
                        if 'y' not in state_str.lower() and self.agents[tid].time_in_phase > min_time:
                            curr_phase = traci.trafficlight.getPhase(tid)
                            n_phases = len(traci.trafficlight.getAllProgramLogics(tid)[0].phases)
                            traci.trafficlight.setPhase(tid, (curr_phase + 1) % n_phases)
                    except:
                        pass

            self.sim.step()
            
            traffic_data = self.sim.get_traffic_data()
            counts = np.array([traffic_data[t]["count"] for t in self.tls_ids], dtype=float)
            total_v = counts.sum()
            mu_new = counts / (total_v + 1e-9) if total_v > 0 else mu
            
            # --- OT Pushing Logic: Distance-based Weighting ---
            target_push = np.copy(target_dist)
            for vid, _ in amb_lanes_info:
                corridor = self.sim.get_upcoming_corridor(vid, look_ahead=1)
                for tid, dist in corridor:
                    if tid in self.tls_ids:
                        idx = self.tls_ids.index(tid)
                        # The closer the ambulance, the less traffic we want (OT PUSH)
                        push_magnitude = max(0.01, 1.0 - (dist / 200.0))
                        target_push[idx] *= (1.0 - push_magnitude)
            
            if target_push.sum() > 0:
                target_push /= target_push.sum()

            if step % 5 == 0:
                _, _, guidance = self.engine.compute_ot_guidance(mu_new, target_push)
            
            for i, tid in enumerate(self.tls_ids):
                next_state = self.get_state(tid, guidance[i])
                states[tid] = next_state

        traci.close()

def main():
    sim = TraCISimulator(SUMO_BINARY, SUMO_CONFIG, NETWORK_FILE)
    graph = TrafficGraph(NETWORK_FILE)
    engine = OTEngine(np.eye(4)) 
    coordinator = HybridCoordinator(sim, graph, engine)
    
    print("\n--- [SHOWCASE] Starting Demonstration with Ambulance ---")
    coordinator.run_epoch(steps=2000, train=False, spawn_amb_interval=400)
    print("\nDemonstration Finished.")

if __name__ == "__main__":
    main()
