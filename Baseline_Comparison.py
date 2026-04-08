import traci
import numpy as np

# --- CONFIGURATION ---
SUMO_BINARY = r"D:\WORK\SUMO\bin\sumo-gui.exe"
SUMO_CONFIG = r"D:\PROJECTS\CCP PJCTS\SEM 4\LynxSpiderWeb\Intersection\Intersection.sumocfg"

def run_baseline(steps=2000):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--waiting-time-memory", "1000"])
    
    tls_ids = traci.trafficlight.getIDList()
    tls_lanes = {tls: list(set(traci.trafficlight.getControlledLanes(tls))) for tls in tls_ids}
    
    halting_history = []
    
    print(f"--- Starting Baseline (Default SUMO) for {steps} steps ---")
    
    for step in range(steps):
        traci.simulationStep()
        
        # Collect halting vehicles globally
        step_halting = 0
        for tls in tls_ids:
            for lane in tls_lanes[tls]:
                step_halting += traci.lane.getLastStepHaltingNumber(lane)
        
        halting_history.append(step_halting)
        
        if step % 200 == 0:
            avg_halting = np.mean(halting_history)
            print(f"Step {step} | Average Halting vehicles (overall): {avg_halting:.2f}")

    traci.close()
    
    final_avg = np.mean(halting_history)
    print("\n" + "="*40)
    print(f"BASELINE FINISHED")
    print(f"Total Average Halting Vehicles: {final_avg:.2f}")
    print("="*40)
    
    return final_avg

if __name__ == "__main__":
    run_baseline()
