from pyannote.audio import Pipeline
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from huggingface_hub import HfApi
from huggingface_hub import login
from pyannote.audio import Pipeline


hf_token = "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"
# login(token=hf_token, add_to_git_credential=True)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
 
# apply pretrained pipeline

diarization = pipeline("nr_chunk7.wav")
 
# # print the result

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s xxxxstop={turn.end:.1f}s speaker_{speaker}")


 