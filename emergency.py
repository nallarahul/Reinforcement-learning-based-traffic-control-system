import traci

class EmergencyHandler:
    def __init__(self, traffic_light_id, lane_to_phase_map):
        self.tl_id = traffic_light_id
        self.lane_to_phase_map = lane_to_phase_map

    def detect_ambulance(self):
        """
        Checks if an ambulance is present in any incoming lane
        """
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) == "ambulance":
                lane_id = traci.vehicle.getLaneID(veh_id)
                if lane_id in self.lane_to_phase_map:
                    return lane_id
        return None

    def apply_priority(self):
        """
        Overrides traffic light to give green to ambulance lane
        """
        lane = self.detect_ambulance()
        if lane is not None:
            phase = self.lane_to_phase_map[lane]
            traci.trafficlight.setPhase(self.tl_id, phase)
            return True
        return False
