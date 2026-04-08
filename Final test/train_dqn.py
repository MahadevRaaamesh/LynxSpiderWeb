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
MODEL_SAVE_PATH = "ot_hybrid_dqn_weights.pt"

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
    def __init__(self, sumo_binary, sumo_config):
        self.sumo_cmd = [sumo_binary, "-c", sumo_config, "--waiting-time-memory", "1000", "--no-step-log", "true"]
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
        # Add epsilon for numerical stability (avoids divide-by-zero if mu has zeros)
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
        self.tls_ids = self.sim.tls_ids
        for tid in self.tls_ids:
            if tid not in self.agents:
                n_lanes = len(self.sim.tls_lanes[tid])
                state_dim = n_lanes + 1 + 1 + 1
                self.agents[tid] = HybridAgent(tid, state_dim)

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

    def save_weights(self, path):
        weights = {tid: agent.policy_net.state_dict() for tid, agent in self.agents.items()}
        torch.save(weights, path)
        print(f"\n[OK] Weights successfully saved to {path}")

    def run_epoch(self, steps=1000, train=True):
        self.sim.start()
        self._init_agents()
        self.tls_ids = self.sim.tls_ids 
        C = self.graph.get_cost_matrix(self.tls_ids)
        self.engine.C = C 
        
        total_wd = 0
        wait_history = []
        
        traffic_data = self.sim.get_traffic_data()
        mu = np.array([traffic_data[t]["count"] for t in self.tls_ids], dtype=float)
        mu = mu / (mu.sum() + 1e-9)
        target_dist = np.ones_like(mu) / (len(mu) + 1e-9)
        _, _, guidance = self.engine.compute_ot_guidance(mu, target_dist)
        
        states = {tid: self.get_state(tid, guidance[i]) for i, tid in enumerate(self.tls_ids)}

        for step in range(steps):
            actions = {}
            for tid in self.tls_ids:
                actions[tid] = self.agents[tid].select_action(states[tid], train)

            for tid in self.tls_ids:
                if actions[tid] == 1: 
                    try:
                        state_str = traci.trafficlight.getRedYellowGreenState(tid)
                        if 'y' not in state_str.lower() and self.agents[tid].time_in_phase > 10:
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
            
            if step % 5 == 0:
                _, wd, guidance = self.engine.compute_ot_guidance(mu_new, target_dist)
                total_wd += wd
            
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
            
            wait_history.append(sum(traffic_data[t]["halting"] for t in self.tls_ids))

            if train and step % 100 == 0:
                for agent in self.agents.values():
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                    agent.epsilon = max(0.05, agent.epsilon * 0.995)

        traci.close()
        return np.mean(wait_history)

def main():
    sim = TraCISimulator(SUMO_BINARY, SUMO_CONFIG)
    graph = TrafficGraph(NETWORK_FILE)
    engine = OTEngine(np.eye(4)) 
    coordinator = HybridCoordinator(sim, graph, engine)
    
    num_episodes = 10
    print(f"\n--- [TRAINING] Starting {num_episodes} Episodes ---")
    for ep in range(num_episodes):
        avg_wait = coordinator.run_epoch(steps=1000, train=True)
        # Use a safe way to get epsilon even if agents dictionary is not yet populated (though it should be after run_epoch)
        eps = next(iter(coordinator.agents.values())).epsilon if coordinator.agents else 1.0
        print(f"Episode {ep+1:02d} | Avg Halting: {avg_wait:.2f} | Epsilon: {eps:.3f}")

    coordinator.save_weights(MODEL_SAVE_PATH)
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
