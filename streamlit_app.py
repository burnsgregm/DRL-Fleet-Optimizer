import streamlit as st
import requests
import numpy as np
import time
import os
import random
import plotly.graph_objects as go

# --- Get Backend URL from Secrets ---
# This MUST be set in your Streamlit Cloud app settings
BACKEND_URL = os.environ.get("BACKEND_URL")

# --- Re-define Simulation Environment ---
# The frontend needs this class to run the simulation.
# The "brain" (model) lives on the backend.
class SimulationEnvironment:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.state = [0, 0]
        self.goal = [random.randint(1, self.grid_size - 1), random.randint(1, self.grid_size - 1)]
        self.step_count = 0
        self.max_steps = self.grid_size * 4
        self.total_reward = 0
        return self.get_json_state()

    def get_json_state(self):
        """Returns a JSON-serializable state for the API."""
        return {
            "agent_x": self.state[0],
            "agent_y": self.state[1],
            "goal_x": self.goal[0],
            "goal_y": self.goal[1],
            "grid_size": self.grid_size
        }

    def step(self, action: int):
        self.step_count += 1
        old_state = list(self.state)

        if action == 0: self.state[0] -= 1 # Up
        elif action == 1: self.state[0] += 1 # Down
        elif action == 2: self.state[1] -= 1 # Left
        elif action == 3: self.state[1] += 1 # Right

        reward = -1
        done = False

        if (self.state[0] < 0 or self.state[0] >= self.grid_size or 
            self.state[1] < 0 or self.state[1] >= self.grid_size):
            reward = -10
            self.state = old_state
        
        if self.state == self.goal:
            reward = 100
            done = True
        
        if self.step_count >= self.max_steps:
            reward = -50
            done = True

        self.total_reward += reward
        return self.get_json_state(), reward, done

# --- Visualization Function ---
def draw_grid(env_state):
    grid_size = env_state['grid_size']
    agent_pos = (env_state['agent_x'], env_state['agent_y'])
    goal_pos = (env_state['goal_x'], env_state['goal_y'])
    
    # Create a 2D array for the grid
    grid = np.zeros((grid_size, grid_size))
    grid[goal_pos] = 0.5  # Goal
    grid[agent_pos] = 1.0 # Agent

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale=[[0, 'white'], [0.5, 'blue'], [1, 'red']],
        showscale=False
    ))
    
    fig.update_layout(
        width=400, height=400,
        margin=dict(l=10, r=10, b=10, t=10),
        xaxis=dict(showgrid=True, tickvals=[], showticklabels=False, linecolor='black'),
        yaxis=dict(showgrid=True, tickvals=[], showticklabels=False, scaleanchor='x', linecolor='black')
    )
    return fig

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="DRL Fleet Optimizer")
st.title("?? DRL Fleet Optimizer")

if not BACKEND_URL:
    st.error("BACKEND_URL secret is not set! Please deploy the FastAPI backend and add its URL as a Streamlit secret.")
    st.stop()
    
st.caption(f"Frontend connected to API backend at: {BACKEND_URL}")

# Initialize session state
if "env" not in st.session_state:
    st.session_state.env = SimulationEnvironment()
    st.session_state.deliveries = 0
    st.session_state.total_steps = 0
    st.session_state.total_fuel_cost = 0
    st.session_state.last_reward = 0
    st.session_state.running = False

env = st.session_state.env

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Simulation Control")
    if st.button("Start / Reset Simulation", use_container_width=True):
        env.reset()
        st.session_state.total_reward = 0
        st.session_state.total_steps = 0
        st.session_state.last_reward = 0
        st.session_state.running = True
        st.success("New simulation started. Agent at (0, 0). New goal set!")

    if st.session_state.running:
        if st.button("Next Step (Call API)", use_container_width=True, type="primary"):
            # 1. Get current state from local sim
            current_state_json = env.get_json_state()
            
            # 2. Call the FastAPI backend for a decision
            try:
                response = requests.post(f"{BACKEND_URL}/get_action", json=current_state_json)
                response.raise_for_status() # Raise error for 4xx/5xx
                action = response.json()['action']
                
                # 3. Apply the action to the local sim
                next_state_json, reward, done = env.step(action)
                st.session_state.total_steps += 1
                st.session_state.total_fuel_cost -= reward if reward < 0 else 0
                st.session_state.last_reward = reward

                if done:
                    if reward == 100:
                        st.session_state.deliveries += 1
                        st.success(f"Delivery Complete! Total Reward: {env.total_reward}")
                    else:
                        st.error(f"Simulation Failed (Timeout). Total Reward: {env.total_reward}")
                    st.session_state.running = False
                    st.button("Start New Simulation", use_container_width=True)
                        
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")
                st.session_state.running = False

    st.header("Live Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Total Deliveries", st.session_state.deliveries)
    m2.metric("Total Steps", st.session_state.total_steps)
    m1.metric("Est. Fuel Cost", f"${st.session_state.total_fuel_cost}")
    m2.metric("Last Reward", st.session_state.last_reward)

with col2:
    st.header("Fleet Position")
    st.plotly_chart(draw_grid(env.get_json_state()), use_container_width=True)

