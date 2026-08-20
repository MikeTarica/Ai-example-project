import streamlit as st
import pandas as pd  # <-- Added to handle structured data inputs
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import time

# 1. Page Configuration
st.set_page_config(page_title="ML Property Predictor", page_icon="🏠", layout="centered")
st.title("🏠 Live Machine Learning & Data Visualizer")
st.write("This app loads a saved Scikit-Learn model from your external drive with optimized features.")

st.markdown("---")

# 2. Load the Saved Model and Mock Dataset for plotting
@st.cache_resource
def load_saved_model_and_data():
    # Load the physically saved model from your external drive
    model = joblib.load('housing_model.pkl')
    
    # We keep the training dataframe reference here just to draw the graph accurately
    raw_data = {
        'SquareFeet': [600, 850, 1200, 1500, 1800, 2200, 2500, 3000, 3500, 4000],
        'Price': [120000, 155000, 200000, 250000, 290000, 360000, 400000, 490000, 560000, 650000]
    }
    df = pd.DataFrame(raw_data)
    return model, df

ml_model, training_df = load_saved_model_and_data()

# 3. User Interface Input
st.subheader("1. Enter Property Characteristics")
user_sqft = st.slider("Square Footage:", min_value=500, max_value=4500, value=1500, step=50)

st.markdown("---")

# 4. Running Predictions & Plotting
st.subheader("2. Model Valuation Analysis")

if st.button("🚀 Calculate AI Valuation"):
    
    # --- FIXED BLOCK: Passing data as a DataFrame to prevent UserWarning ---
    input_data = pd.DataFrame([[user_sqft]], columns=['SquareFeet'])
    
    # Predict using the loaded frozen file (clean and warning-free!)
    predicted_array = ml_model.predict(input_data)
    prediction = float(predicted_array[0])
    # ----------------------------------------------------------------------
    
    # Display Metrics Boxes
    st.success("### Valuation Generated Successfully!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Selected Size", value=f"{user_sqft:,} sq ft")
    with col2:
        st.metric(label="Model Predicted Price", value=f"${prediction:,.2f}")
        
    st.markdown("---")
    st.subheader("3. Visualizing the Regression Line")
    
    # Create the Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Plot 1: The original training data points (Scatter plot)
    ax.scatter(training_df['SquareFeet'], training_df['Price'], color='blue', label='Actual Data Points', alpha=0.7)
    
    # Plot 2: The trendline the AI discovered (Line plot)
    line_X_raw = np.linspace(500, 4500, 100).reshape(-1, 1)
    # We structure the line plot data as a DataFrame too to keep it completely warning-free
    line_X_df = pd.DataFrame(line_X_raw, columns=['SquareFeet'])
    line_y = ml_model.predict(line_X_df)
    ax.plot(line_X_raw, line_y, color='red', linestyle='--', label='AI Regression Line')
    
    # Plot 3: Highlight the user's specific selection
    ax.scatter([user_sqft], [prediction], color='green', s=150, zorder=5, label='Your Selection')
    
    # Chart Styling
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Pass the matplotlib figure to Streamlit
    st.pyplot(fig)
    
    st.balloons()


    # 1. Page Config
st.set_page_config(page_title="AI Private Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Private Local Chatbot Workspace")
st.write("This application retains context across conversations using session state storage.")
st.markdown("---")

# 2. Initialize Chat Memory (Session State)
# If this is the first time opening the page, create an empty list to store messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your private AI assistant. How can I help you build today?"}
    ]

# 3. Display Chat History Matrix
# Loop through our recorded memory and draw them inside speech bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle Live User Chat Input
# This creates the standard fixed chat input bar at the bottom of the screen
if user_prompt := st.chat_input("Type your message here..."):
    
    # Step A: Display the user's message live on screen immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Step B: Record the user's message permanently into our session history array
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Step C: Generate the AI response bubble
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # --- PLACEHOLDER FOR COGNITIVE REASONING ENGINE ---
        # Right now this is a hardcoded mock rule engine response.
        # This is exactly where we will later connect your live web scraper or API models!
        mock_reply = f"I received your message: '{user_prompt}'. My workspace engine is ready to connect an open-source LLM or data web scraping pipeline here."
        # ---------------------------------------------------
        
        # Simulating live streaming text token-by-token for a premium feel
        for token in mock_reply.split(" "):
            full_response += token + " "
            time.sleep(0.06) # Tiny delay to mimic thinking
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
    # Step D: Append the assistant's finalized answer to our session history array
    st.session_state.messages.append({"role": "assistant", "content": full_response})