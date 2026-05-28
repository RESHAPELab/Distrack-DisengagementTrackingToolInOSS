import pandas as pd
import numpy as np

def apply_increasing_noise(df, column_name, noise_scale=0.01):
    # 1. Identify streaks of 1s
    group_id = (df[column_name] != df[column_name].shift()).cumsum()
    
    # 2. Count consecutive occurrences
    streak_count = df.groupby(group_id).cumcount()
    df['streak'] = np.where(df[column_name] == 1, streak_count, 0)
    
    # 3. Generate Positive-Only Noise
    # We use absolute value so it only ever adds to the 1.0
    raw_noise = np.abs(np.random.normal(0, noise_scale, len(df)))
    
    # 4. Calculate Adjusted Value
    # Value = 1 + (streak * positive_noise)
    # This ensures that as streak grows, the potential addition grows
    df['adjusted_value'] = np.where(
        df[column_name] == 1,
        1.0 + (df['streak'] * raw_noise),
        0.0
    )
    
    return df

# Example with your logic
data = {'signal': [0, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 0, 0,0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]}
df = pd.DataFrame(data)
result = apply_increasing_noise(df, 'signal', noise_scale=0.05)
print(result)