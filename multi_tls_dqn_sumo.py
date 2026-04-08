# multi_tls_dqn_sumo.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opti
import random
from collections import deque
import traci

# ===================== CONFIG =====================
SUMO_BINARY = "D://WORK//SUMO//bin//sumo-gui.exe"
SUMO_CFG = "D://PROJECTS//CCP PJCTS//SEM 4//LynxSpiderWeb//Intersection//Intersection.sumocfg"
MODEL_PATH = "multi_tls_dqn.pt"

MAX_STEPS_PER_EP = 2000
EPISODES = 50
REPLAY_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
STATE_HORIZON = 1   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True
# =================================================

def start_sumo():
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--no-step-log",
        "--waiting-time-memory", "1000"
    ])

# -------- DQN NETWORK --------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------- REPLAY BUFFER --------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch):
        batch = random.sample(self.buffer, batch)
        s,a,r,s2,d = map(np.stack, zip(*batch))
        return s,a,r,s2,d

    def __len__(self):
        return len(self.buffer)

# -------- ENV HELPERS --------
def get_tls_ids():
    return traci.trafficlight.getIDList()

def get_lanes_for_tls(tls):
    return traci.trafficlight.getControlledLanes(tls)

def get_num_phases(tls):
    logic = traci.trafficlight.getAllProgramLogics(tls)[0]
    return len(logic.phases)

def get_state_for_tls(tls):
    lanes = get_lanes_for_tls(tls)
    q = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
    phase = traci.trafficlight.getPhase(tls)
    return np.array(q + [phase], dtype=np.float32)

def global_state(tls_ids):
    states = [get_state_for_tls(t) for t in tls_ids]
    return np.concatenate(states)

def step_all_tls(actions, tls_ids, phase_counts):
    for i, tls in enumerate(tls_ids):
        if actions[i] == 1:
            cur = traci.trafficlight.getPhase(tls)
            traci.trafficlight.setPhase(tls, (cur + 1) % phase_counts[i])

    traci.simulationStep()

def compute_reward(tls_ids):
    total_wait = 0.0
    total_queue = 0.0
    for tls in tls_ids:
        for v in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(v)
        for l in get_lanes_for_tls(tls):
            total_queue += traci.lane.getLastStepHaltingNumber(l)
    return -(total_wait + 0.5 * total_queue)

# -------- MAIN TRAINING LOOP --------
def train():
    start_sumo()
    tls_ids = get_tls_ids()
    phase_counts = [get_num_phases(t) for t in tls_ids]
    num_tls = len(tls_ids)

    state_dim = len(global_state(tls_ids))
    action_dim = 2 * num_tls  # 0/1 per TLS

    policy = DQN(state_dim, action_dim).to(device)
    target = DQN(state_dim, action_dim).to(device)
    target.load_state_dict(policy.state_dict())

    optim = opti.Adam(policy.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_SIZE)

    epsilon = EPS_START

    for ep in range(EPISODES):
        
        traci.close()
        start_sumo()

        tls_ids = get_tls_ids()
        s = global_state(tls_ids)
        done = False
        total_reward = 0

        for t in range(MAX_STEPS_PER_EP):
            if ep == 0 and t == 0:
                print("Policy is on:", next(policy.parameters()).device)
            # epsilon-greedy
            if random.random() < epsilon:
                a = np.random.randint(0, 2, size=num_tls)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device)   
                    q = policy(s_t)                                           
                    a = q.view(num_tls, 2).argmax(dim=1).cpu().numpy()      

            step_all_tls(a, tls_ids, phase_counts)
            s2 = global_state(tls_ids)
            r = compute_reward(tls_ids)
            done = traci.simulation.getMinExpectedNumber() == 0

            buffer.push(s, a, r, s2, done)
            s = s2
            total_reward += r

            if len(buffer) >= BATCH_SIZE:
                bs, ba, br, bs2, bd = buffer.sample(BATCH_SIZE)

                bs_t = torch.tensor(bs, dtype=torch.float32,device=device)
                ba_t = torch.tensor(ba, dtype=torch.long,device=device)
                br_t = torch.tensor(br, dtype=torch.float32,device=device)
                bs2_t = torch.tensor(bs2, dtype=torch.float32,device=device)
                bd_t = torch.tensor(bd, dtype=torch.float32,device=device)

                q = policy(bs_t).view(BATCH_SIZE, num_tls, 2)
                q = q.gather(2, ba_t.unsqueeze(-1)).squeeze(-1).sum(dim=1)

                with torch.no_grad():
                    q_next = target(bs2_t).view(BATCH_SIZE, num_tls, 2).max(dim=2)[0].sum(dim=1)
                    q_target = br_t + GAMMA * (1 - bd_t) * q_next

                loss = nn.MSELoss()(q, q_target)
                optim.zero_grad()
                loss.backward()
                optim.step()

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

        print(f"Episode {ep} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f}")

    torch.save(policy.state_dict(), MODEL_PATH)
    traci.close()

if __name__ == "__main__":
    train()
