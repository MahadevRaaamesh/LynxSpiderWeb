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
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SUMO_BINARY = r"D:\WORK\SUMO\bin\sumo-gui.exe"
SUMO_CONFIG = r"D:\LEARNING\Traffic System\Intersection\Intersection.sumocfg"
NETWORK_FILE = r"D:\LEARNING\Traffic System\Intersection\Intersection.net.xml"
MODEL_SAVE_PATH = "ot_hybrid_dqn_ambulance.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LAYER 1: NEURAL NETWORK (DQN) ---
class DQN(nn.Module):
    """
    Standard MLP for Q-value estimation.
    State: [Local Queues, Phase ID, Time in Phase, OT Guidance Signal]
    Actions: [Stay (0), Switch (1)]
    """
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
    def __init__(self, sumo_binary, sumo_config):
        self.sumo_cmd = [sumo_binary, "-c", sumo_config, "--waiting-time-memory", "1000"]
        self.tls_ids = []
        self.tls_lanes = {}
        self.edges = []

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
        if len(self.edges) < 2: return
        
        # Check if this exact ambulance is already in the network
        try:
            if amb_id in traci.vehicle.getIDList():
                return False
        except:
            pass
            
        # Give it an unmistakable appearance
        try:
            if "ambulance_type" not in traci.vehicletype.getIDList():
                traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_type")
                traci.vehicletype.setVehicleClass("ambulance_type", "emergency")
                traci.vehicletype.setColor("ambulance_type", (255, 0, 0, 255))
                traci.vehicletype.setLength("ambulance_type", 6.5)
                traci.vehicletype.setMaxSpeed("ambulance_type", 30.0)
                traci.vehicletype.setShapeClass("ambulance_type", "emergency")
        except Exception as e:
            pass
            
        for _ in range(20):  # Try finding a valid route
            src = random.choice(self.edges)
            dst = random.choice(self.edges)
            if src == dst: continue
            
            try:
                # Need valid routing info, ignore output if it fails to avoid console spam
                route = traci.simulation.findRoute(src, dst)
                if route and route.edges and len(route.edges) > 1:
                    route_id = f"route_{amb_id}_{random.randint(10000, 99999)}"
                    traci.route.add(route_id, route.edges)
                    traci.vehicle.add(amb_id, route_id, typeID="ambulance_type")
                    
                    # Aggressive driving for ambulance
                    traci.vehicle.setSpeedFactor(amb_id, 1.5)
                    traci.vehicle.setLaneChangeMode(amb_id, 0b100000101001)  # Aggressive but relatively safe lane changes
                    
                    print(f"\033[91m🚨 EMERGENCY: Ambulance {amb_id} spawned at {src} to {dst}\033[0m")
                    return True
            except traci.TraCIException:
                 # TraCIException is raised when no route is found, ignore
                pass
            except Exception as e:
                pass
        return False

    def get_ambulance_lanes(self):
        # Extract the lane ID for every active ambulance
        lanes = []
        try:
            for vid in traci.vehicle.getIDList():
                vtype = traci.vehicle.getTypeID(vid)
                if vtype == "ambulance_type" or "amb" in vid:
                    current_lane = traci.vehicle.getLaneID(vid)
                    if current_lane and not current_lane.startswith(":"):
                        lanes.append((vid, current_lane))
        except:
            pass
        return lanes

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
    def __init__(self, cost_matrix, reg=0.1):
        self.C = cost_matrix
        self.reg = reg

    def compute_ot_guidance(self, mu, target):
        """Returns (gamma, WD, guidance_vector)"""
        gamma = ot.sinkhorn(mu, target, self.C, self.reg)
        wd = np.sum(gamma * self.C)
        guidance = target - mu # Positive means "needs more capacity/less traffic"
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
        self.epsilon = 1.0
        self.gamma = 0.95
        self.last_phase = -1
        self.time_in_phase = 0

    def select_action(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(DEVICE)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.memory) < batch_size: return
        s, a, r, s2, d = self.memory.sample(batch_size)
        
        s = torch.FloatTensor(s).to(DEVICE)
        a = torch.LongTensor(a).to(DEVICE)
        r = torch.FloatTensor(r).to(DEVICE)
        s2 = torch.FloatTensor(s2).to(DEVICE)
        d = torch.FloatTensor(d).to(DEVICE)

        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(s2).max(1)[0]
        expected_q = r + (1 - d) * self.gamma * q_next

        loss = nn.MSELoss()(q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class HybridCoordinator:
    def __init__(self, sim, graph, engine):
        self.sim = sim
        self.graph = graph
        self.engine = engine
        self.agents = {}
        self.tls_ids = []

    def _init_agents(self):
        """Initialize agents after simulation starts and TLS IDs are known"""
        self.tls_ids = self.sim.tls_ids
        for tid in self.tls_ids:
            if tid not in self.agents:
                n_lanes = len(self.sim.tls_lanes[tid])
                state_dim = n_lanes + 1 + 1 + 1
                self.agents[tid] = HybridAgent(tid, state_dim)

    def get_state(self, tid, guidance_val):
        """Constructs a non-blind state vector"""
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

    def run_epoch(self, steps=1000, train=True, spawn_amb_interval=300):
        self.sim.start()
        self._init_agents()
        self.tls_ids = self.sim.tls_ids 
        C = self.graph.get_cost_matrix(self.tls_ids)
        self.engine.C = C 
        
        total_wd = 0
        total_wait = 0
        wait_history = []
        
        # Initial state
        traffic_data = self.sim.get_traffic_data()
        mu = np.array([traffic_data[t]["count"] for t in self.tls_ids], dtype=float)
        mu = mu / (mu.sum() + 1e-9)
        target_dist = np.ones_like(mu) / len(mu)
        _, _, guidance = self.engine.compute_ot_guidance(mu, target_dist)
        
        states = {tid: self.get_state(tid, guidance[i]) for i, tid in enumerate(self.tls_ids)}

        ambulance_counter = 0

        for step in range(steps):
            # 1. Spawn ambulances
            if not train and step > 0 and step % spawn_amb_interval == 0:
                self.sim.spawn_ambulance(f"amb_{ambulance_counter}")
                ambulance_counter += 1

            # 2. Track Ambulances
            amb_lanes_info = self.sim.get_ambulance_lanes()
            active_amb_edges = set()
            for v, l in amb_lanes_info:
                try:
                    active_amb_edges.add(traci.lane.getEdgeID(l))
                except:
                    pass
            
            actions = {}
            active_amb_tls = set()

            # 3. Decision Phase
            for tid in self.tls_ids:
                # Normal AI Action
                agent_action = self.agents[tid].select_action(states[tid], train)
                
                # --- PREEMPTION LOGIC ---
                amb_override = False
                
                try:
                    state_str = traci.trafficlight.getRedYellowGreenState(tid)
                    links = traci.trafficlight.getControlledLinks(tid)
                    
                    # Check if an ambulance is approaching this intersection
                    amb_indices = []
                    for i, link_group in enumerate(links):
                        if not link_group: continue
                        in_lane = link_group[0][0]
                        in_edge = traci.lane.getEdgeID(in_lane)
                        
                        if in_edge in active_amb_edges:
                            amb_indices.append(i)
                    
                    if amb_indices:
                        amb_override = True
                        active_amb_tls.add(tid)
                        
                        # See if current phase is green for ambulance
                        is_green = False
                        for idx in amb_indices:
                            if idx < len(state_str) and state_str[idx].lower() == 'g':
                                is_green = True
                                break
                                
                        if is_green:
                            # Hold Green
                            agent_action = 0 
                        else:
                            # Force Switch
                            agent_action = 1
                except Exception as e:
                    pass
                
                actions[tid] = agent_action

            # 4. Apply Actions
            for tid in self.tls_ids:
                if actions[tid] == 1: # Switch
                    try:
                        state_str = traci.trafficlight.getRedYellowGreenState(tid)
                        amb_override = tid in active_amb_tls
                        
                        # Ambulances drop the minimum phase duration to 3 seconds safely
                        min_time = 3 if amb_override else 10
                        
                        if 'y' not in state_str.lower() and self.agents[tid].time_in_phase > min_time:
                            curr_phase = traci.trafficlight.getPhase(tid)
                            n_phases = len(traci.trafficlight.getAllProgramLogics(tid)[0].phases)
                            traci.trafficlight.setPhase(tid, (curr_phase + 1) % n_phases)
                    except:
                        pass

            self.sim.step()
            
            # Real-time traffic metrics
            traffic_data = self.sim.get_traffic_data()
            counts = np.array([traffic_data[t]["count"] for t in self.tls_ids], dtype=float)
            total_v = counts.sum()
            mu_new = counts / (total_v + 1e-9) if total_v > 0 else mu
            
            # OT computation
            if step % 5 == 0:
                _, wd, guidance = self.engine.compute_ot_guidance(mu_new, target_dist)
                total_wd += wd
            
            # Reward and next state
            for i, tid in enumerate(self.tls_ids):
                next_state = self.get_state(tid, guidance[i])
                
                local_halting = traffic_data[tid]["halting"]
                ot_penalty = abs(guidance[i]) * 10 
                switch_cost = 2.0 if actions[tid] == 1 else 0.0
                
                reward = -(local_halting * 0.1) - ot_penalty - switch_cost
                
                if train:
                    self.agents[tid].memory.push(states[tid], actions[tid], reward, next_state, False)
                    self.agents[tid].update()
                
                states[tid] = next_state
                total_wait += local_halting
            
            wait_history.append(sum(traffic_data[t]["halting"] for t in self.tls_ids))

            if train and step % 100 == 0:
                for agent in self.agents.values():
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                    agent.epsilon = max(0.05, agent.epsilon * 0.995)

            if step > 0 and step % 200 == 0:
                avg_halt = np.mean(wait_history)
                print(f"Step {step} | WD: {total_wd/(step/5+1):.4f} | Avg Halting: {avg_halt:.2f}")

        traci.close()
        final_avg = np.mean(wait_history)
        return total_wd, final_avg

def main():
    sim = TraCISimulator(SUMO_BINARY, SUMO_CONFIG)
    graph = TrafficGraph(NETWORK_FILE)
    engine = OTEngine(np.eye(4)) 
    
    coordinator = HybridCoordinator(sim, graph, engine)
    
    print("\n--- Phase 1: Training Hybrid Agent (3 Episodes without ambulance) ---")
    for ep in range(3):
        wd, wait = coordinator.run_epoch(steps=1000, train=True)
        print(f"Episode {ep} finished. Avg Halting: {wait:.2f}")

    print("\n--- Phase 2: Final Demonstration (2000 steps with ambulance preemption) ---")
    wd, final_avg = coordinator.run_epoch(steps=2000, train=False, spawn_amb_interval=250)
    print("\n" + "="*40)
    print(f"HYBRID AMBULANCE SIMULATION FINISHED")
    print(f"Final Average Halting Vehicles: {final_avg:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
