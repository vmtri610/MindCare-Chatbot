import streamlit as st

# Header with logo and title
st.title("🧠 MindCare Chatbot")
st.markdown("**Support for mental health consultation and monitoring based on DSM-5 standards**")

# Introduction section
st.header("📖 Introduction")
st.markdown("""
Welcome to the **Mental Healthcare System**, an application powered by artificial intelligence (AI) and the **LlamaIndex** library to create an intelligent mental health assistant. The system helps with:
- **Mental health consultation**: Engage in natural conversations to relieve emotions and address psychological concerns.
- **Preliminary analysis and diagnosis**: Assess mental health conditions based on DSM-5 standards, covering common disorders like anxiety and depression.
- **Progress tracking**: Store interaction history and provide recommendations to improve mental health over time.

The system is designed with a user-friendly interface and ensures **data privacy**, complying with personal information protection regulations.
""")

# Usage guide section
st.header("🚀 Usage Guide")
st.markdown("""
To start using the system, follow these steps:

1. **Interact with the AI assistant**:
   - Access the chat interface.
   - Converse naturally with the AI assistant as you would with a friend. Share your emotions, symptoms, or any issues you're facing.
   - The assistant will collect information to evaluate your mental health condition.

2. **Receive assessment and recommendations**:
   - Once sufficient information is provided or you choose to end the conversation, the assistant will summarize and provide a preliminary mental health assessment (categorized into four levels: Poor, Average, Normal, Good).
   - The system will offer specific recommendations, such as relaxation exercises or healthy habits to enhance mental well-being.

3. **Track progress**:
   - The system stores your interaction history and mental health metrics.
   - You can review your mental health progress through charts and statistics in the "Tracking" section.

4. **Use regularly**:
   - For optimal results, use the application frequently to monitor and improve your mental health.
""")

# Additional information section
st.header("ℹ️ Additional Information")
st.markdown("""
- **Data source**: The system uses the "Diagnostic Criteria for Mental Disorders based on DSM-5" (a condensed 106-page version) as the basis for analysis and diagnosis.
""")

# Footer
st.markdown("---")
st.markdown("© 2025 All rights reserved.")
