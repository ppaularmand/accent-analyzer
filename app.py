import streamlit as st
import random
import time

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

            # Generate simulated results
            random.seed(hash(video_url) % 10000)

            # Pick a random accent with higher probability for some common ones
            weights = [0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]
            accent = random.choices(ACCENT_CLASSES, weights=weights)[0]

            # Generate a confidence score between 60-95%
            confidence = round(random.uniform(60, 95), 2)

            # Generate random scores for other accents
            remaining = 100 - confidence
            other_scores = []
            for _ in range(len(ACCENT_CLASSES) - 1):
                if remaining > 0:
                    score = min(round(random.uniform(0, remaining), 2), remaining)
                    other_scores.append(score)
                    remaining -= score
                else:
                    other_scores.append(0)

            # Create all scores dictionary
            all_scores = {accent: confidence}
            for a, s in zip([a for a in ACCENT_CLASSES if a != accent], other_scores):
                all_scores[a] = s

            # Generate explanation
            explanations = {
                "American": "Speaker demonstrates typical American rhotic 'r' sounds with a standard cadence consistent with General American accent patterns.",
                "British": "Speaker exhibits non-rhotic pronunciation and typical British vowel qualities consistent with RP (Received Pronunciation) patterns.",
                "Australian": "Speaker shows characteristic Australian vowel raising and intonation patterns with distinctive rising terminals.",
                "Indian": "Speaker demonstrates syllable-timed rhythm (vs stress-timed) with dental consonants typical of Indian English varieties.",
                "Canadian": "Speaker shows Canadian raising in diphthongs and mild rhoticity with characteristic Canadian vowel patterns.",
                "Irish": "Speaker exhibits Irish-typical intonation patterns and vowel articulation with characteristic consonant softening.",
                "Scottish": "Speaker demonstrates Scottish vowel system and rhotic pronunciation with characteristic Scottish intonation.",
                "South African": "Speaker shows South African vowel system with characteristic intonation patterns and typical pronunciation of consonants."
            }
            explanation = explanations.get(accent, "No detailed explanation available.")

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