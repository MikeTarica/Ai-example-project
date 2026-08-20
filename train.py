import pandas as pd
import numpy as np
import time

# 1. Our Dataset (Scaled down slightly so the numbers are easy to read)
X = np.array([6, 8, 12, 15, 18, 22])       # Size in hundreds of sq ft (6 = 600)
Y = np.array([120, 155, 200, 250, 290, 360]) # Price in thousands (120 = $120k)

# 2. Start with a completely random guess for our model's line
slope = 0.0
intercept = 0.0
learning_rate = 0.001 # How big of a step the AI takes to correct mistakes

print("--- STARTING THE LEARNING PROCESS ---")
print(f"Initial Guess -> Slope: {slope}, Intercept: {intercept}\n")

# 3. The Training Loop (Watch it update!)
for epoch in range(1, 101):
    # Calculate predictions based on current slope/intercept
    predictions = (slope * X) + intercept
    
    # Calculate the error (how far off the AI is)
    errors = predictions - Y
    
    # Calculate the "gradient" (which direction to turn the knob to fix the line)
    slope_gradient = (2 / len(X)) * np.sum(errors * X)
    intercept_gradient = (2 / len(X)) * np.sum(errors)
    
    # UPDATE STEP: Adjust the parameters slightly
    slope -= learning_rate * slope_gradient
    intercept -= learning_rate * intercept_gradient
    
    # Print out the updates every 20 steps so you can see it learn
    if epoch % 20 == 0 or epoch == 1:
        current_loss = np.mean(errors ** 2)
        print(f"Epoch {epoch}:")
        print(f"   👉 Model updated Slope to: {slope:.4f}")
        print(f"   👉 Model updated Intercept to: {intercept:.4f}")
        print(f"   ❌ Average Error Margin (Loss): {current_loss:.2f}\n")