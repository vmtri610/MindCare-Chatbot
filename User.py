import streamlit as st
import json
import pandas as pd

# File path for scores
SCORES_FILE = "data/user_storage/scores.json"


# Load data from JSON file
def load_scores():
    try:
        with open(SCORES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return []


# Prepare data for chart
def prepare_data_for_chart(scores):
    if not scores:
        return pd.DataFrame()
    df = pd.DataFrame(scores)
    df['Time'] = pd.to_datetime(df['Time'])
    # Format Time column to show only date and time
    df['Time'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


# Display detailed information for a selected timestamp
def display_details(scores, selected_time):
    for entry in scores:
        entry_time = pd.to_datetime(entry['Time'])
        if entry_time == selected_time:
            st.subheader(f"Details for {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Score:** {entry['Score']}")
            st.write(f"**Level:** {entry['Level']}")
            st.write(f"**Content:** {entry['Content']}")
            st.write(f"**Total Guess:** {entry['Total guess']}")
            break
    else:
        st.write("No data available for the selected time.")


# Main function for User page
st.title("🧠 Mental Health Tracking")

# Load scores data
scores = load_scores()

# Prepare data for chart
df = prepare_data_for_chart(scores)

# Display chart
st.header("📊 Score History")
if not df.empty:
    chart_data = df[['Time', 'Score']].set_index('Time')
    # Customize the line chart to mimic the image style
    st.line_chart(chart_data, use_container_width=True)


else:
    st.warning("No data available to display.")

# Timestamp selection
st.header("📅 View Details by Time")
if df.empty:
    st.warning("No data available to select.")
else:
    timestamps = [pd.to_datetime(entry['Time']) for entry in scores]
    # Add current time to options
    selected_time = st.selectbox("Select a timestamp", options=timestamps,
                                 format_func=lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    display_details(scores, selected_time)
