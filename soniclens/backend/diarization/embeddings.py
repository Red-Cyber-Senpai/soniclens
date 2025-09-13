# Placeholder embeddings helper — returns dummy vectors for MVP
def get_embeddings_for_wav_segments(wav_path, segments):
    import numpy as np
    return np.random.rand(len(segments), 256)
