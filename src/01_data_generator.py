import numpy as np
import pandas as pd
import os

def generate_crop_seasons(num_seasons=1000, days_per_season=120, seed=42):
    np.random.seed(seed)
    all_seasons_data = []

    for season_id in range(num_seasons):
        base_temp = 20 + 10 * np.sin(np.linspace(0, np.pi, days_per_season))
        temp = base_temp + np.random.normal(0, 2, days_per_season)
        precip = np.random.gamma(shape=0.5, scale=5.0, size=days_per_season)
        precip = np.where(precip < 2.0, 0, precip) 
        
        irrigation_applied = np.random.choice([0, 10, 20], size=days_per_season, p=[0.7, 0.2, 0.1])
        nitrogen_applied = np.random.choice([0, 5, 15], size=days_per_season, p=[0.85, 0.1, 0.05])
        
        soil_moisture = np.zeros(days_per_season)
        soil_nitrogen = np.zeros(days_per_season)
        biomass = np.zeros(days_per_season)
        nitrogen_leaching = np.zeros(days_per_season)
        
        current_moisture, current_nitrogen, current_biomass = 30.0, 50.0, 0.0   
        
        for day in range(days_per_season):
            evapo = max(1.0, temp[day] * 0.2)
            current_moisture = np.clip(current_moisture + precip[day] + irrigation_applied[day] - evapo, 10, 100) 
            soil_moisture[day] = current_moisture
            
            leach_rate = 0.1 if current_moisture > 80 else 0.02
            leached_N = current_nitrogen * leach_rate
            nitrogen_leaching[day] = leached_N
            
            current_nitrogen = np.clip(current_nitrogen + nitrogen_applied[day] - leached_N, 0, 200)
            soil_nitrogen[day] = current_nitrogen
            
            moisture_factor = np.clip((current_moisture - 20) / 80, 0, 1)
            nitrogen_factor = np.clip(current_nitrogen / 100, 0, 1)
            temp_factor = np.clip((temp[day] - 10) / 20, 0, 1)
            
            daily_growth = 150 * moisture_factor * nitrogen_factor * temp_factor * (1 - current_biomass/10000) if current_moisture > 10 and temp[day] > 5 else 0
            current_biomass += daily_growth
            biomass[day] = current_biomass
            
        all_seasons_data.append(pd.DataFrame({
            'season_id': season_id, 'day': np.arange(days_per_season), 'temperature_c': temp,
            'precipitation_mm': precip, 'irrigation_mm': irrigation_applied, 'nitrogen_kgha': nitrogen_applied,
            'soil_moisture_mm': soil_moisture, 'soil_nitrogen_kgha': soil_nitrogen,
            'nitrogen_leached_kgha': nitrogen_leaching, 'biomass_kgha': biomass
        }))
        
    return pd.concat(all_seasons_data, ignore_index=True)

if __name__ == "__main__":
    print("Generating synthetic agricultural telemetry data...")
    df_telemetry = generate_crop_seasons(num_seasons=1000)
    
    os.makedirs('data', exist_ok=True)
    file_path = 'data/synthetic_telemetry.csv'
    df_telemetry.to_csv(file_path, index=False)
    
    print(f"Data generated successfully! Saved to {file_path}")
    print(f"Dataset Shape: {df_telemetry.shape}")