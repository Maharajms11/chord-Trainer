# Guitar Chord Tone Trainer

Standalone Streamlit app for learning guitar chord tones.

## Features
- Common chord qualities (major, minor, sus, diminished, sevenths)
- Blues-focused chords (dominant 7, 9, 13, 7#9)
- Interactive fretboard map
- I-IV-V blues helper
- Quiz mode for chord-tone recall

## Run locally
1. Create/activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Render
1. Push this folder to a GitHub repository.
2. In Render, choose **New +** -> **Blueprint**.
3. Select the repository. Render will detect `render.yaml`.
4. Confirm and create the service.

Render uses:
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --browser.gatherUsageStats false`
