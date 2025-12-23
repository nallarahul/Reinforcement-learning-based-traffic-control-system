from __future__ import absolute_import, print_function
import os, sys, time, optparse, random, serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# SUMO setup
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci


# ---------- Utility functions ----------
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


# ---------- Dueling Network ----------
class DuelingModel(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DuelingModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        # Value and Advantage streams
        self.value = nn.Linear(fc2_dims, 1)
        self.advantage = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value(x)
        advantage = self.advantage(x)

        # Combine value and advantage into Q-values
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals


# ---------- Agent ----------
class Agent:
    def __init__(
        self, gamma, epsilon, lr, input_dims, fc1_dims, fc2_dims, batch_size,
        n_actions, junctions, max_memory_size=100000, epsilon_dec=5e-4, epsilon_end=0.05
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.replace_target = 1000  # target update frequency
        self.learn_step_counter = 0

        # Two networks: Evaluation and Target
        self.Q_eval = DuelingModel(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_target = DuelingModel(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # Replay memory per junction
        self.memory = {
            j: {
                "state": np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "new_state": np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "reward": np.zeros(max_memory_size, dtype=np.float32),
                "action": np.zeros(max_memory_size, dtype=np.int32),
                "done": np.zeros(max_memory_size, dtype=bool),
                "mem_cntr": 0,
            }
            for j in junctions
        }

    def store_transition(self, state, state_, action, reward, done, junction):
        mem = self.memory[junction]
        index = mem["mem_cntr"] % self.max_mem
        mem["state"][index] = state
        mem["new_state"][index] = state_
        mem["reward"][index] = reward
        mem["action"][index] = action
        mem["done"][index] = done
        mem["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                actions = self.Q_eval(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self, junction):
        mem = self.memory[junction]
        if mem["mem_cntr"] < self.batch_size:
            return

        max_mem = min(mem["mem_cntr"], self.max_mem)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(mem["state"][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(mem["new_state"][batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(mem["reward"][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(mem["done"][batch]).to(self.Q_eval.device)
        action_batch = torch.tensor(mem["action"][batch]).to(self.Q_eval.device)

        # Current Q-values
        q_eval = self.Q_eval(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Double DQN: use Q_eval to select action, Q_target to evaluate it
        next_actions = torch.argmax(self.Q_eval(new_state_batch), dim=1)
        q_next = self.Q_target(new_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next
        loss = self.Q_eval.loss(q_eval, q_target.detach())

        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        # Update target network occasionally
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def save(self, model_name):
        os.makedirs("models", exist_ok=True)
        torch.save(self.Q_eval.state_dict(), f"models/{model_name}.bin")


# ---------- Training Loop ----------
# ---------- Training Loop ----------
def run(train=True, model_name="model_d3qn", epochs=50, steps=500, ard=False):
    if ard:
        arduino = serial.Serial(port="COM4", baudrate=9600, timeout=0.1)
        def write_read(x):
            arduino.write(bytes(x, "utf-8"))
            time.sleep(0.05)
            return arduino.readline()

    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg"])
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    traci.close()

    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.001,
        input_dims=4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=128,
        n_actions=4,
        junctions=junction_numbers,
    )

    if not train:
        model_path = f"models/{model_name}.bin"
        if os.path.exists(model_path):
            agent.Q_eval.load_state_dict(torch.load(model_path))
            agent.Q_target.load_state_dict(agent.Q_eval.state_dict())
            print(f"Loaded pretrained model: {model_path}")
        else:
            print(f"Model {model_path} not found, running untrained.")

    total_time_list = []
    best_time = np.inf

    for e in range(epochs):
        binary = "sumo" if train else "sumo-gui"
        traci.start([checkBinary(binary), "-c", "configuration.sumocfg"])
        print(f"Epoch {e+1}/{epochs}")

        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        step, total_time, min_duration = 0, 0, 5
        traffic_lights_time, prev_action, prev_state = {}, {}, {}

        # NEW: Metric trackers
        total_vehicles_passed = 0
        total_queue_length = 0
        time_steps_counted = 0

        for jn, j in enumerate(all_junctions):
            traffic_lights_time[j] = 0
            prev_action[jn] = 0
            prev_state[jn] = [0] * 4

        while step <= steps:
            traci.simulationStep()
            time_steps_counted += 1  # NEW

            for jn, j in enumerate(all_junctions):
                controlled_lanes = traci.trafficlight.getControlledLanes(j)
                waiting_time = get_waiting_time(controlled_lanes)
                total_time += waiting_time

                # NEW: Queue length tracking
                queue_length = sum(traci.lane.getLastStepHaltingNumber(l) for l in controlled_lanes)
                total_queue_length += queue_length

                # NEW: Vehicle throughput
                total_vehicles_passed += len(traci.simulation.getDepartedIDList())

                if traffic_lights_time[j] == 0:
                    vehicles = get_vehicle_numbers(controlled_lanes)
                    state_ = list(vehicles.values())
                    reward = -waiting_time

                    agent.store_transition(prev_state[jn], state_, prev_action[jn], reward, (step == steps), jn)
                    action = agent.choose_action(state_)
                    prev_action[jn] = action
                    prev_state[jn] = state_

                    phaseDuration(j, 6, select_lane[action][0])
                    phaseDuration(j, min_duration + 10, select_lane[action][1])
                    traffic_lights_time[j] = min_duration + 10

                    if train:
                        agent.learn(jn)
                else:
                    traffic_lights_time[j] -= 1

            step += 1

        # --- Epoch Metrics ---
        avg_waiting_time_per_vehicle = total_time / max(total_vehicles_passed, 1)
        avg_queue_length = total_queue_length / max(time_steps_counted, 1)

        print(f"Total waiting time: {total_time}")
        print(f"Average waiting time per vehicle: {avg_waiting_time_per_vehicle:.2f}")
        print(f"Total vehicles passed (throughput): {total_vehicles_passed}")
        print(f"Average queue length: {avg_queue_length:.2f}")
        print("------------------------------------------------------")

        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                agent.save(model_name)
                print(f"Saved improved model to models/{model_name}.bin")

        traci.close()

    if train:
        plt.plot(total_time_list)
        plt.xlabel("Epochs")
        plt.ylabel("Total Waiting Time")
        plt.title("Training Progress (Dueling Double DQN)")
        plt.savefig(f"plots/time_vs_epoch_{model_name}.png")
        plt.show()



def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-m", dest="model_name", type="string", default="model_d3qn")
    optParser.add_option("--train", action="store_true", default=False)
    optParser.add_option("-e", dest="epochs", type="int", default=50)
    optParser.add_option("-s", dest="steps", type="int", default=500)
    optParser.add_option("--ard", action="store_true", default=False)
    return optParser.parse_args()[0]


if __name__ == "__main__":
    opts = get_options()
    run(train=opts.train, model_name=opts.model_name, epochs=opts.epochs, steps=opts.steps, ard=opts.ard)
