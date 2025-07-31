import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web server
import matplotlib.pyplot as plt
from scipy import signal
import os
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import json

class ActiveNoiseCancellation:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path):
        """Load audio file and return audio data and sample rate"""
        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f"Loaded audio: {file_path}")
            print(f"Duration: {len(audio_data)/sr:.2f} seconds")
            print(f"Sample rate: {sr} Hz")
            return audio_data, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None
    
    def generate_anti_sound(self, audio_data):
        """Generate anti-sound by inverting the phase (180° shift)"""
        anti_sound = -audio_data
        return anti_sound
    
    def apply_anc(self, original_audio, anti_sound, mix_ratio=0.8):
        """Apply ANC by mixing original audio with anti-sound"""

        min_length = min(len(original_audio), len(anti_sound))
        original_audio = original_audio[:min_length]
        anti_sound = anti_sound[:min_length]
        

        cleaned_audio = original_audio + (mix_ratio * anti_sound)
        

        max_amplitude = np.max(np.abs(cleaned_audio))
        if max_amplitude > 1.0:
            cleaned_audio = cleaned_audio / max_amplitude * 0.95
            print(f"Warning: Audio normalized due to high amplitude ({max_amplitude:.2f})")
        
        # Ensure no NaN or infinite values
        cleaned_audio = np.nan_to_num(cleaned_audio, nan=0.0, posinf=0.95, neginf=-0.95)
        
        return cleaned_audio
    
    def save_audio(self, audio_data, output_path, sample_rate=None):
        """Save audio data to file"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Ensure audio data is valid before saving
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.95, neginf=-0.95)
            
            # Clamp values to valid range for audio
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            sf.write(output_path, audio_data, sample_rate)
            print(f"Audio saved: {output_path}")
        except Exception as e:
            print(f"Error saving audio file: {e}")
    
    def plot_waveforms_web(self, original, anti_sound, cleaned):
        """Generate plot for web display and return as base64 string"""
        try:
            time = np.linspace(0, len(original)/self.sample_rate, len(original))
            
            plt.figure(figsize=(15, 10))
            
            # Original audio
            plt.subplot(3, 1, 1)
            plt.plot(time, original, 'b-', alpha=0.7)
            plt.title('Original Audio', fontsize=14)
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # Anti-sound
            plt.subplot(3, 1, 2)
            plt.plot(time, anti_sound, 'r-', alpha=0.7)
            plt.title('Anti-Sound (180° Phase Shift)', fontsize=14)
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # Cleaned audio
            plt.subplot(3, 1, 3)
            plt.plot(time, cleaned, 'g-', alpha=0.7)
            plt.title('Cleaned Audio (After ANC)', fontsize=14)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
        except Exception as e:
            print(f"Error generating waveform plot: {e}")
            plt.close()
            return None
    
    def analyze_frequency_spectrum_web(self, original, cleaned):
        """Generate frequency analysis plot for web display"""
        try:
            # Compute FFT
            original_fft = np.fft.fft(original)
            cleaned_fft = np.fft.fft(cleaned)
            

            freqs = np.fft.fftfreq(len(original), 1/self.sample_rate)
            

            positive_freqs = freqs[:len(freqs)//2]
            original_magnitude = np.abs(original_fft[:len(original_fft)//2])
            cleaned_magnitude = np.abs(cleaned_fft[:len(cleaned_fft)//2])
            

            epsilon = 1e-10
            original_magnitude = np.maximum(original_magnitude, epsilon)
            cleaned_magnitude = np.maximum(cleaned_magnitude, epsilon)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.semilogy(positive_freqs, original_magnitude, 'b-', alpha=0.7, label='Original')
            plt.semilogy(positive_freqs, cleaned_magnitude, 'g-', alpha=0.7, label='Cleaned')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('Frequency Spectrum Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, self.sample_rate//4)
            
            # Noise reduction analysis with proper handling of edge cases
            plt.subplot(1, 2, 2)
            noise_reduction = 20 * np.log10(cleaned_magnitude / original_magnitude)
            
            # Clamp extreme values for better visualization
            noise_reduction = np.clip(noise_reduction, -100, 100)
            
            plt.plot(positive_freqs, noise_reduction, 'purple', alpha=0.7)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Noise Reduction (dB)')
            plt.title('Noise Reduction by Frequency')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, self.sample_rate//4)
            plt.ylim(-50, 50)  
            
            plt.tight_layout()
            

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
        except Exception as e:
            print(f"Error generating frequency plot: {e}")
            plt.close()
            return None
    
    def process_audio_web(self, input_file, mix_ratio=0.8):
        """Process audio for web interface and return results"""
        # Load audio
        audio_data, sr = self.load_audio(input_file)
        if audio_data is None:
            return None
        
        # Generate anti-sound
        anti_sound = self.generate_anti_sound(audio_data)
        
        # Apply ANC
        cleaned_audio = self.apply_anc(audio_data, anti_sound, mix_ratio)
        
        # Calculate statistics with proper validation
        original_rms = np.sqrt(np.mean(audio_data**2))
        cleaned_rms = np.sqrt(np.mean(cleaned_audio**2))
        
        # Ensure valid numerical values
        if not np.isfinite(original_rms) or original_rms == 0:
            original_rms = 1e-10
        if not np.isfinite(cleaned_rms):
            cleaned_rms = 1e-10
            
        noise_reduction_db = 20 * np.log10(cleaned_rms / original_rms)
        
        # Sanitize all values to ensure they're JSON serializable
        def sanitize_value(val):
            if not np.isfinite(val) or np.isnan(val):
                return 0.0
            return float(val)
        
        # Generate plots
        waveform_plot = self.plot_waveforms_web(audio_data, anti_sound, cleaned_audio)
        frequency_plot = self.analyze_frequency_spectrum_web(audio_data, cleaned_audio)
        
        # Save audio files
        os.makedirs('static/output', exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        anti_sound_path = f'static/output/{base_name}_anti_sound.wav'
        cleaned_audio_path = f'static/output/{base_name}_cleaned.wav'
        
        self.save_audio(anti_sound, anti_sound_path)
        self.save_audio(cleaned_audio, cleaned_audio_path)
        
        return {
            'success': True,
            'original_rms': sanitize_value(original_rms),
            'cleaned_rms': sanitize_value(cleaned_rms),
            'noise_reduction_db': sanitize_value(noise_reduction_db),
            'duration': sanitize_value(len(audio_data) / sr),
            'sample_rate': int(sr),
            'waveform_plot': waveform_plot,
            'frequency_plot': frequency_plot,
            'anti_sound_file': anti_sound_path.replace('static/', ''),
            'cleaned_audio_file': cleaned_audio_path.replace('static/', ''),
            'filename': base_name
        }


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'


os.makedirs('uploads', exist_ok=True)
os.makedirs('static/output', exist_ok=True)


anc = ActiveNoiseCancellation()

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_directory(directory_path):
    """Remove all files in a directory."""
    if not os.path.isdir(directory_path):
        return
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Clear previous output files
        clear_directory('static/output')
        clear_directory(app.config['UPLOAD_FOLDER'])

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Get mix ratio from form
        mix_ratio = float(request.form.get('mix_ratio', 0.8))
        mix_ratio = max(0.0, min(1.0, mix_ratio))  # Clamp between 0 and 1
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process audio
        result = anc.process_audio_web(file_path, mix_ratio)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        if result is None:
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Validate result before returning
        try:
            json.dumps(result)  # Test if result is JSON serializable
        except (TypeError, ValueError) as e:
            print(f"JSON serialization error: {e}")
            return jsonify({'error': 'Data processing error - invalid numerical values'}), 500
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid mix ratio value: {str(e)}'}), 400
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(f'static/{filename}', as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

def main():
    """Main function to run the web server"""
    print("Starting Active Noise Cancellation Web Server...")
    print("=" * 50)
    print("Server will be available at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=8080)

if __name__ == "__main__":
    main()