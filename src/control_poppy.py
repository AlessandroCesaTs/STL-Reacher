import socket
import time
from stable_baselines3 import PPO
import pickle
import numpy as np

model=PPO.load('outputs/fixed_hard_const/model.zip')

with open('outputs/fixed_hard_const/setting.pkl', 'rb') as f:
            setting = pickle.load(f)

goal=setting['goal']
initial_pose=setting['initial_pose']
avoid=setting['avoid']

print(f"goal is {goal}",flush=True)


initial_action={'m1':np.degrees(initial_pose[0]).item(),'m2':np.degrees(initial_pose[1]).item(),'m3':np.degrees(initial_pose[2]).item(),
                'm4':np.degrees(initial_pose[3]).item(),'m5':np.degrees(initial_pose[4]).item(),'m6':np.degrees(initial_pose[5]).item()}

print(f"initial action {initial_action}")


def convert_actions(new_actions):
    """
    Convert the model's predicted actions (in radians) to a dictionary with motor positions in degrees.
    
    Parameters:
    new_actions (np.array): Array of motor actions in radians (length = 6)
    
    Returns:
    dict: Dictionary with motor positions in degrees
    """
    # Convert radians to degrees
    
    # Create the dictionary for motor positions
    motor_positions_deg = {
        'm1': np.degrees(new_actions[0]).item(),
        'm2': np.degrees(new_actions[1]).item(),
        'm3': np.degrees(new_actions[2]).item(),
        'm4': np.degrees(new_actions[3]).item(),
        'm5': np.degrees(new_actions[4]).item(),
        'm6': np.degrees(new_actions[5]).item()
    }
    
    return motor_positions_deg


def send_actions_to_robot(actions, connection):
    actions_str = str(actions)  # Convert the dictionary to a string
    print(f"Sending actions: {actions_str}")  # Debug print
    connection.sendall(actions_str.encode('utf-8'))

def process_motor_positions(motor_positions):
    """
    Generate new motor position dictionary based on received motor positions.
    """
    # Example of processing motor positions and adjusting them

    obs_list = []
    
    # Flatten motor positions
    for i in range(6):
        obs_list.append(np.radians(motor_positions[f'm{i+1}_position']).item())  # Assume each motor position is a list
    
    # Flatten motor speeds
    for i in range(6):
        obs_list.append(np.radians(motor_positions[f'm{i+1}_speed']).item())  # Assuming speed is a single value (not a list)
    
    # Flatten end effector position (assuming it's a list)
    obs_list.extend(motor_positions['end_effector'])
    
    # Convert the list to a 1D NumPy array
    obs = np.array(obs_list)

    new_actions=model.predict(obs)

    return convert_actions(new_actions[0])

def get_distance_from_goal(position):
     return np.linalg.norm(goal-position)


if __name__ == "__main__":
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('169.254.86.254', 12345))  # Use your Raspberry Pi IP address

    try:
        send_actions_to_robot(initial_action, client_socket)
        distance_from_goal=np.inf
        while distance_from_goal>0.02:  # Loop to receive positions and send actions multiple times
            # Receive motor positions from the Raspberry Pi
            data = client_socket.recv(1024).decode('utf-8')
            motor_positions = eval(data)  # Convert the string back to a dictionary
            print(f"Motor positions received: {motor_positions}")  # Debug print
            distance_from_goal=get_distance_from_goal(motor_positions['end_effector'])
            print(f"distance_from_goal {distance_from_goal}")        
            # Generate new actions based on received motor positions
            new_actions = process_motor_positions(motor_positions)

            # Send the new actions to the Raspberry Pi
            send_actions_to_robot(new_actions, client_socket)

            #time.sleep(0.5)  # Optional: wait before sending the next set of actions
    finally:
        client_socket.close()  # Close the socket connection
