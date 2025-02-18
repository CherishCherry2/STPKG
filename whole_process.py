import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_mapping(mapping_file):
    action_to_id = {}
    id_to_action = {}
    with open(mapping_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                action_id = int(parts[0])
                action_name = parts[1]
                action_to_id[action_name] = action_id
                id_to_action[action_id] = action_name
    return action_to_id, id_to_action


def process_text_file(file_path, action_to_id):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    action_sequence = []
    previous_action = None
    for line in lines:
        action_id = action_to_id[line]
        if action_id != previous_action:
            action_sequence.append(action_id)
            previous_action = action_id


    return action_sequence


def process_directory(directory_path, mapping_file):
    action_to_id, id_to_action = load_mapping(mapping_file)
    num_actions = len(action_to_id)
    transition_counts = np.zeros((num_actions, num_actions), dtype=int)

    txt_files = sorted(glob.glob(os.path.join(directory_path, '*.txt')))
    for file_path in txt_files:
        action_sequence = process_text_file(file_path, action_to_id)
        for i in range(len(action_sequence) - 1):
            from_action = action_sequence[i]
            to_action = action_sequence[i + 1]
            transition_counts[from_action][to_action] += 1

    transition_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probabilities = np.divide(transition_counts, transition_sums,
                                         out=np.zeros_like(transition_counts, dtype=float),
                                         where=transition_sums != 0)

    return transition_probabilities, id_to_action


def plot_transition_matrix(transition_probabilities, id_to_action):
    num_actions = len(id_to_action)
    action_names = [id_to_action[i] for i in range(num_actions)]

    fig, ax = plt.subplots(figsize=(12, 12)) 
    cax = ax.matshow(transition_probabilities, cmap='viridis')

    plt.xticks(range(num_actions), action_names, rotation=90)
    plt.yticks(range(num_actions), action_names)

    ax.set_xticks(range(num_actions))
    ax.set_xticklabels(action_names, rotation=45, ha='right')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_yticks(range(num_actions))
    ax.set_yticklabels(action_names)

    fig.colorbar(cax)

    plt.xlabel('To Action')
    plt.ylabel('From Action')
    plt.title('Transition Probability Matrix')

    plt.tight_layout()  
    plt.savefig('transition_matrix.tif', bbox_inches='tight')  
    plt.show()


