import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import joblib
import os

class AgriTwinEnv(gym.Env):
    def __init__(self, surrogate_model, weather_data):
        super().__init__()
        self.model = surrogate_model
        self.weather_data = weather_data
        self.current_day = 0
        self.days_per_season = 120
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.state = np.array([30.0, 50.0, 0.0])
        
        season_id = np.random.choice(self.weather_data['season_id'].unique())
        self.season_weather = self.weather_data[self.weather_data['season_id'] == season_id].reset_index(drop=True)
        return self._get_obs(), {}
        
    def _get_obs(self):
        temp = self.season_weather.loc[self.current_day, 'temperature_c']
        precip = self.season_weather.loc[self.current_day, 'precipitation_mm']
        obs = np.concatenate([self.state, [temp, precip, self.current_day / self.days_per_season]])
        return obs.astype(np.float32)
        
    def step(self, action):
        irrigation = np.clip((action[0] + 1) * 10, 0, 20) 
        nitrogen = np.clip((action[1] + 1) * 7.5, 0, 15)  
        
        temp = self.season_weather.loc[self.current_day, 'temperature_c']
        precip = self.season_weather.loc[self.current_day, 'precipitation_mm']
        
        features = np.array([[self.state[0], self.state[1], self.state[2], irrigation, nitrogen, temp, precip]])
        next_state = self.model.predict(features)[0]
        next_state = np.clip(next_state, 0, None)
        
        biomass_gain = next_state[2] - self.state[2]
        cost = (irrigation * 0.1) + (nitrogen * 0.5) 
        reward = biomass_gain - cost
        
        self.state = next_state
        self.current_day += 1
        
        terminated = self.current_day >= self.days_per_season - 1
        truncated = False
        
        return self._get_obs(), float(reward), terminated, truncated, {}

def train_and_test_rl():
    print("Loading Surrogate Model and Data...")
    try:
        surrogate = joblib.load('models/xgboost_surrogate.pkl')
        df = pd.read_csv('data/synthetic_telemetry.csv')
    except FileNotFoundError:
        print("Error: Could not find model or data. Please run scripts 01 and 02 first.")
        return

    env = AgriTwinEnv(surrogate_model=surrogate, weather_data=df)

    print("Training AI Agent (PPO)... This will take about 10-20 seconds...")
    agent = PPO("MlpPolicy", env, verbose=0, n_steps=1024)
    agent.learn(total_timesteps=100000)
    
    os.makedirs('models', exist_ok=True)
    agent.save("models/ppo_farm_agent")
    print("AI Training Complete! Model saved to models/ppo_farm_agent.zip\n")

    print("Testing the trained AI on a new farming season...")
    obs, _ = env.reset()
    biomass_history, irrigation_history, nitrogen_history = [], [], []

    for _ in range(120):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        irrigation_history.append(np.clip((action[0] + 1) * 10, 0, 20))
        nitrogen_history.append(np.clip((action[1] + 1) * 7.5, 0, 15))
        biomass_history.append(obs[2])
        
        if terminated or truncated:
            break

    print("Generating performance plot...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(biomass_history, color='purple', label='Crop Biomass (Yield)', linewidth=3)
    ax2.bar(range(len(irrigation_history)), irrigation_history, color='blue', alpha=0.4, label='AI Irrigation (mm)')
    ax2.bar(range(len(nitrogen_history)), nitrogen_history, color='green', alpha=0.4, label='AI Nitrogen (kg/ha)', bottom=irrigation_history)

    ax1.set_xlabel('Day of Season')
    ax1.set_ylabel('Biomass (kg/ha)', color='purple', fontweight='bold')
    ax2.set_ylabel('Resources Applied by AI', color='black', fontweight='bold')
    plt.title('Digital Twin: AI Farm Manager Optimizing Yield vs. Resource Usage')
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.grid(True, alpha=0.3)
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/latest_run.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    train_and_test_rl()