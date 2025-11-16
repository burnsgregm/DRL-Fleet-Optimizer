import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Model & App Loading ---
MODEL_FILE = "drl_fleet_model.h5"

# Load the pre-trained Keras model
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print("DRL model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI(title="DRL Fleet Optimizer API")

# --- Pydantic Models for API ---
class FleetState(BaseModel):
    """Defines the state payload we expect from the frontend."""
    agent_x: int
    agent_y: int
    goal_x: int
    goal_y: int
    grid_size: int

class ActionResponse(BaseModel):
    action: int

# --- Helper Function ---
def get_state_representation(state: FleetState) -> np.ndarray:
    """Converts the JSON state into the numpy array the model expects."""
    grid = np.zeros((state.grid_size, state.grid_size))
    grid[state.agent_x, state.agent_y] = 1.0  # Agent position
    grid[state.goal_x, state.goal_y] = 0.5  # Goal position
    return grid.flatten().reshape(1, state.grid_size * state.grid_size)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "DRL Fleet Optimizer API is online."}

@app.post("/get_action", response_model=ActionResponse)
def get_action(state: FleetState):
    if model is None:
        raise fastapi.HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Convert JSON state to Numpy array
    state_array = get_state_representation(state)
    
    # 2. Get Q-values from the model (Exploit mode)
    q_values = model.predict(state_array, verbose=0)
    
    # 3. Get the best action (index with the highest Q-value)
    action = int(np.argmax(q_values[0]))
    
    return {"action": action}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
