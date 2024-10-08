import yt_dlp
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pyannote.audio import Pipeline
import os
import torch

# Step 1: Download audio from YouTube using yt-dlp
def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return 'downloaded_audio.wav'

# Step 2: Split audio into chunks based on 500 ms of silence, min 6 sec, max 18 sec
def split_audio_based_on_silence(input_file, min_chunk_size=6000, max_chunk_size=18000, silence_len=500):
    audio = AudioSegment.from_file(input_file)
    chunk_dir = 'qualified_chunks'  # Save only the qualified chunks
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)

    chunk_index = 0
    chunk_paths = []
    remaining_audio = audio  # Start with the full audio

    while len(remaining_audio) > 0:
        # Split the remaining audio based on 500 ms of silence
        chunks = split_on_silence(remaining_audio, min_silence_len=silence_len, silence_thresh=-40)

        if len(chunks) == 0:
            # If no silence is found, break out of the loop to avoid infinite loops
            break

        # Track the total processed duration to move forward
        total_processed_duration = 0

        for chunk in chunks:
            chunk_duration = len(chunk)
            total_processed_duration += chunk_duration  # Accumulate processed duration

            if min_chunk_size <= chunk_duration <= max_chunk_size:
                # Save the valid chunk with the prefix chunk_test
                chunk_filename = f'{chunk_dir}/chunk_test_{chunk_index}.wav'
                chunk.export(chunk_filename, format='wav')
                print(f"Saved qualified chunk: {chunk_filename} | Duration: {chunk_duration / 1000:.2f} seconds")
                chunk_paths.append(chunk_filename)
                chunk_index += 1
            elif chunk_duration > max_chunk_size:
                # If chunk is too long, split it further into max_chunk_size pieces
                for i in range(0, len(chunk), max_chunk_size):
                    sub_chunk = chunk[i:i+max_chunk_size]
                    sub_chunk_duration = len(sub_chunk)
                    if sub_chunk_duration >= min_chunk_size:
                        sub_chunk_filename = f'{chunk_dir}/chunk_test_{chunk_index}.wav'
                        sub_chunk.export(sub_chunk_filename, format='wav')
                        print(f"Saved sub-chunk: {sub_chunk_filename} | Duration: {sub_chunk_duration / 1000:.2f} seconds")
                        chunk_paths.append(sub_chunk_filename)
                        chunk_index += 1

        # Remove the processed portion of the audio by updating remaining_audio
        remaining_audio = remaining_audio[total_processed_duration:]

    print(f"Completed splitting the audio. Total qualified chunks: {chunk_index}")
    return chunk_paths

# Step 3: Perform speaker diarization and filter chunks with only one speaker
def analyze_and_filter_chunks(chunks, pipeline):
    filtered_chunks = []
    
    for chunk_file in chunks:
        diarization = pipeline(chunk_file)
        num_speakers = len(set([speaker for _, _, speaker in diarization.itertracks(yield_label=True)]))
        
        if num_speakers == 1:  # Check if only one speaker is present
            filtered_chunks.append(chunk_file)
    
    return filtered_chunks

# Step 4: Save filtered chunks into a separate directory
def save_filtered_chunks(filtered_chunks):
    filtered_chunks_dir = 'filtered_chunks'
    if not os.path.exists(filtered_chunks_dir):
        os.makedirs(filtered_chunks_dir)
    
    for chunk_file in filtered_chunks:
        chunk_audio = AudioSegment.from_file(chunk_file)
        chunk_name = os.path.basename(chunk_file)
        output_path = os.path.join(filtered_chunks_dir, chunk_name)
        
        # Export the filtered chunk to the new directory
        chunk_audio.export(output_path, format='wav')
        print(f'Saved filtered chunk: {output_path}')

# Step 5: Calculate total duration and total voice duration for each filtered chunk
def calculate_durations(filtered_chunks, model, get_speech_timestamps, read_audio, sampling_rate=16000):
    chunk_durations = []  # List to store actual and voice durations for each chunk

    for chunk_file in filtered_chunks:
        # Load the chunk with pydub to get its duration in milliseconds
        chunk_audio = AudioSegment.from_file(chunk_file)
        actual_duration = len(chunk_audio) / 1000  # Convert milliseconds to seconds

        # Load chunk audio for Silero VAD processing
        wav = read_audio(chunk_file, sampling_rate=sampling_rate)
        
        # Get the speech timestamps using Silero VAD
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
        
        # Calculate voice duration using Silero VAD
        voice_duration = 0
        for segment in speech_timestamps:
            start_sample = segment['start']
            end_sample = segment['end']
            segment_duration = (end_sample - start_sample) / sampling_rate  # Convert samples to seconds
            voice_duration += segment_duration

        # Store the actual and voice duration for this chunk
        chunk_durations.append({
            'chunk': chunk_file,
            'actual_duration': actual_duration,
            'voice_duration': voice_duration
        })

    return chunk_durations

# Main function to process audio from YouTube, split into dynamic chunks, and analyze speakers
def process_audio_from_youtube(youtube_url, hf_token):
    # Step 1: Download YouTube audio
    audio_path = download_audio(youtube_url)
    
    # Step 2: Split the audio into chunks based on 500 ms of silence
    chunks = split_audio_based_on_silence(audio_path)

    # Step 3: Load Hugging Face Pyannote speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    # Step 4: Perform diarization and filter chunks with only one speaker
    filtered_chunks = analyze_and_filter_chunks(chunks, pipeline)
    
    # Step 5: Save the filtered chunks to a new directory
    save_filtered_chunks(filtered_chunks)

    # Step 6: Load Silero VAD model and calculate total voice duration
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    (get_speech_timestamps, _, read_audio, *_) = utils

    # Step 7: Calculate both total chunk duration and total voice duration for each chunk
    chunk_durations = calculate_durations(filtered_chunks, model, get_speech_timestamps, read_audio)

    # Step 8: Print individual chunk durations
    for chunk_info in chunk_durations:
        print(f"Chunk: {chunk_info['chunk']}")
        print(f"  Actual Duration: {chunk_info['actual_duration']:.2f} seconds")
        print(f"  Voice Duration (using Silero VAD): {chunk_info['voice_duration']:.2f} seconds")
        print("")

# Example usage
if __name__ == '__main__':
    youtube_url = 'https://www.youtube.com/watch?v=ZJDJZXIGMXI&t=17s'  # Replace with your YouTube URL
    hf_token = "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"  # Replace with your Hugging Face token
    
    process_audio_from_youtube(youtube_url, hf_token)
