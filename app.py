from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch

from dqn_agent import DQNAgent
from main import Config

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Loading the trained model
MODEL_PATH = "./agents/model_2000.pt"

agent = None  # this will hold our DQNAgent once loaded
config = Config()

agent = DQNAgent(config)
agent.dqn_net.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))
agent.dqn_net.eval()

@app.route("/get_action", methods=["POST"])
def get_action():
    state = request.json['state']
    epsilon = request.json.get('epsilon', 0.1)
    action = agent.get_action(np.array(state), epsilon)
    return jsonify({'action': action})

@app.route("/get_q_values", methods=["POST"])
def get_q_values():
    state = request.json['state']
    q_values = agent.get_q_values(np.array(state))
    return jsonify({'q_values': q_values.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)