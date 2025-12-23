from __future__ import absolute_import, print_function
import os, sys, time, optparse, random, serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# --- SUMO PATH SETUP ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci


# ------------------------------
# SUMO helper functions
# ------------------------------
def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


# ------------------------------
# Neural Network Model
# ------------------------------
class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


# ------------------------------
# DDQN Agent
# ------------------------------
class DDQNAgent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05,
        replace_target=100,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = replace_target

        self.Q_eval = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_target = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.update_target_network()

        self.memory = {}
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros((self.max_mem, input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_mem, input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=bool),
                "mem_cntr": 0,
            }

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, state, new_state, action, reward, done, junction):
        idx = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][idx] = state
        self.memory[junction]["new_state_memory"][idx] = new_state
        self.memory[junction]["reward_memory"][idx] = reward
        self.memory[junction]["action_memory"][idx] = action
        self.memory[junction]["terminal_memory"][idx] = done
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                actions = self.Q_eval(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self, junction):
        mem_count = self.memory[junction]["mem_cntr"]
        if mem_count < self.batch_size:
            return

        batch_indices = np.random.choice(
            min(mem_count, self.max_mem), self.batch_size, replace=False
        )
        states = torch.tensor(self.memory[junction]["state_memory"][batch_indices], dtype=torch.float).to(self.Q_eval.device)
        actions = torch.tensor(self.memory[junction]["action_memory"][batch_indices], dtype=torch.long).to(self.Q_eval.device)
        rewards = torch.tensor(self.memory[junction]["reward_memory"][batch_indices], dtype=torch.float).to(self.Q_eval.device)
        next_states = torch.tensor(self.memory[junction]["new_state_memory"][batch_indices], dtype=torch.float).to(self.Q_eval.device)
        dones = torch.tensor(self.memory[junction]["terminal_memory"][batch_indices], dtype=torch.bool).to(self.Q_eval.device)

        # --- DDQN target ---
        q_eval = self.Q_eval(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_eval_next = self.Q_eval(next_states)
        q_target_next = self.Q_target(next_states)

        max_actions = torch.argmax(q_eval_next, dim=1)
        q_target_values = q_target_next.gather(1, max_actions.unsqueeze(1)).squeeze(1)
        q_target_values[dones] = 0.0
        q_target = rewards + self.gamma * q_target_values

        loss = self.Q_eval.loss(q_eval, q_target)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        if self.iter_cntr % self.replace_target == 0:
            self.update_target_network()

        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def save(self, model_name):
        os.makedirs("models", exist_ok=True)
        torch.save(self.Q_eval.state_dict(), f"models/{model_name}.bin")

    def load(self, model_name):
        path = f"models/{model_name}.bin"
        if os.path.exists(path):
            self.Q_eval.load_state_dict(torch.load(path, map_location=self.Q_eval.device))
            self.update_target_network()
            print(f"Loaded DDQN model from {path}")
        else:
            print(f"Model {path} not found.")


# ------------------------------
# Run simulation
# ------------------------------
def run(train=True, model_name="model_ddqn", epochs=200, steps=500, ard=False):
    if ard:
        arduino = serial.Serial(port="COM4", baudrate=9600, timeout=0.1)
        def write_read(x):
            arduino.write(bytes(x, "utf-8"))
            time.sleep(0.05)
            data = arduino.readline()
            return data

    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    traci.close()

    agent = DDQNAgent(
        gamma=0.99,
        epsilon=0.1,
        lr=0.001,
        input_dims=4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=512,
        n_actions=4,
        junctions=junction_numbers,
        replace_target=200,
    )

    if not train:
        agent.load(model_name)

    total_time_list = []
    best_time = np.inf

    for e in range(epochs):
        traci.start(
            [checkBinary("sumo-gui" if not train else "sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
        )
        print(f"Epoch {e+1}/{epochs}")

        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        total_time = 0
        traffic_lights_time = {}
        prev_wait_time = {}
        prev_action = {}
        prev_vehicles = {}

        # NEW: Additional metric trackers
        total_vehicles_passed = 0
        total_queue_length = 0
        time_steps_counted = 0

        for jn, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[jn] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles[jn] = [0] * 4

        step = 0
        while step <= steps:
            traci.simulationStep()
            time_steps_counted += 1  # NEW

            # NEW: Count vehicles passed and queue length
            total_vehicles_passed += len(traci.simulation.getDepartedIDList())
            queue_length_epoch = 0

            for jn, junction in enumerate(all_junctions):
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controlled_lanes)
                total_time += waiting_time

                # NEW: Compute queue length
                queue_length = sum(traci.lane.getLastStepHaltingNumber(l) for l in controlled_lanes)
                queue_length_epoch += queue_length

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controlled_lanes)
                    reward = -waiting_time
                    state_ = list(vehicles_per_lane.values())
                    state = prev_vehicles[jn]
                    prev_vehicles[jn] = state_

                    agent.store_transition(state, state_, prev_action[jn], reward, (step==steps), jn)
                    action = agent.choose_action(state_)
                    prev_action[jn] = action

                    phaseDuration(junction, 6, select_lane[action][0])
                    phaseDuration(junction, 15, select_lane[action][1])

                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        write_read(ph)

                    traffic_lights_time[junction] = 15
                    if train:
                        agent.learn(jn)
                else:
                    traffic_lights_time[junction] -= 1

            total_queue_length += queue_length_epoch  # NEW
            step += 1

        # --- Epoch Metrics ---
        avg_waiting_time_per_vehicle = total_time / max(total_vehicles_passed, 1)
        avg_queue_length = total_queue_length / max(time_steps_counted, 1)

        print(f"Total waiting time this epoch: {total_time}")
        print(f"Average waiting time per vehicle: {avg_waiting_time_per_vehicle:.2f}")
        print(f"Total vehicles passed (throughput): {total_vehicles_passed}")
        print(f"Average queue length: {avg_queue_length:.2f}")
        print("------------------------------------------------------")

        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            agent.save(model_name)
            print(f"Saved improved model to models/{model_name}.bin")

        traci.close()
        if not train:
            break

    if train:
        plt.plot(range(len(total_time_list)), total_time_list)
        plt.xlabel("Epochs")
        plt.ylabel("Total Waiting Time")
        plt.title("DDQN Performance Over Epochs")
        plt.savefig(f"plots/time_vs_epoch_{model_name}.png")
        plt.show()


# ------------------------------
# CLI arguments
# ------------------------------
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-m", dest="model_name", type="string", default="model_ddqn", help="model name")
    optParser.add_option("--train", action="store_true", default=False, help="train or test mode")
    optParser.add_option("-e", dest="epochs", type="int", default=50, help="Number of epochs")
    optParser.add_option("-s", dest="steps", type="int", default=500, help="Number of steps")
    optParser.add_option("--ard", action="store_true", default=False, help="Connect Arduino")
    return optParser.parse_args()[0]


if __name__ == "__main__":
    opts = get_options()
    run(train=opts.train, model_name=opts.model_name, epochs=opts.epochs, steps=opts.steps, ard=opts.ard)
