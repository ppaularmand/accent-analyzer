import streamlit as st
import random
import time
import subprocess
import sys
import os

st.set_page_config(page_title="English Accent Analyzer (Simplified)", page_icon="🎙️")

st.title("English Accent Analyzer")
st.markdown("""
This is a simplified version of the accent analyzer tool for testing purposes.
It simulates the analysis without requiring heavy dependencies.
""")

# Define accent classes
ACCENT_CLASSES = [
    "American", "British", "Australian", "Indian",
    "Canadian", "Irish", "Scottish", "South African"
]

# Input form
st.subheader("Video URL")
video_url = st.text_input(
    "Enter a video URL (YouTube, Loom, or direct MP4 link)",
    placeholder="https://www.youtube.com/watch?v=example"
)

# Process button
if st.button("Analyze Accent", type="primary"):
    if not video_url:
        st.error("Please enter a video URL")
    else:
        with st.spinner("Processing video... (this may take a minute)"):
            # Simulate processing time
            time.sleep(3)
            try:
                # Get the absolute path to accent_analyzer.py
                analyzer_path = os.path.abspath("accent_analyzer.py")
                print(f"Running analyzer script: {analyzer_path}")

                # Call the accent_analyzer.py script using subprocess
                result = subprocess.run(
                    [sys.executable, analyzer_path, video_url],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Extract the output from the subprocess
                output = result.stdout
                error_output = result.stderr
                print(f"Script output: {output}")
                print(f"Script error output: {error_output}")

                lines = output.strip().split('\n')
                if len(lines) >= 2:
                    accent = lines[0].split(':')[-1].strip()
                    confidence_str = lines[1].split(':')[-1].strip().replace('%', '')
                    confidence = float(confidence_str)
                else:
                    accent = "Unknown"
                    confidence = 0

                explanation = "Simulated accent analysis."  # Add a placeholder
                all_scores = {a: round(random.uniform(5, 30), 2) for a in ACCENT_CLASSES}
                all_scores[accent] = round(confidence, 2)

            except subprocess.CalledProcessError as e:
                # Handle errors from the accent_analyzer.py script
                st.error(f"Error analyzing accent: {e.stderr}")
                accent = "Error"
                confidence = 0
                explanation = "Analysis failed."
                all_scores = {}

            # Display results
            st.success("Analysis complete!")

            # Create columns for layout
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Results")
                st.metric("Accent", accent)
                st.metric("Confidence", f"{confidence}%")

                # Display explanation
                st.subheader("Explanation")
                st.info(explanation)

            with col2:
                # Create bar chart of all scores
                st.subheader("Accent Probability Distribution")

                # Sort by values
                sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                sorted_labels = [item[0] for item in sorted_items]
                sorted_values = [item[1] for item in sorted_items]

                # Create bar chart
                st.bar_chart({
                    'Accent': sorted_labels,
                    'Confidence': sorted_values
                })

                # Add download button for the results
                results = {
                    "accent": accent,
                    "confidence": confidence,
                    "explanation": explanation,
                    "all_scores": all_scores
                }

                st.download_button(
                    label="Download Results (JSON)",
                    data=str(results),
                    file_name="accent_analysis.json",
                    mime="application/json"
                )

# Add some information about supported accents
st.subheader("Supported Accent Classifications")
st.markdown("""
The system can currently detect the following accent types:
- American
- British
- Australian
- Indian
- Canadian
- Irish
- Scottish
- South African
""")

# Add footer with disclaimer
st.markdown("---")
st.caption("""
**Disclaimer**: This is a simplified demonstration tool for REM Waste's technical assessment.
The actual implementation would use a fine-tuned model for accent classification.
""")