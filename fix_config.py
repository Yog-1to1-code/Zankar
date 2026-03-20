import json

path = '/Users/apple/Downloads/AI_Models/whisper-large-v3-turbo/config.json'

with open(path, 'r') as f:
    config = json.load(f)

# Native MLX dimensions mapping from Hugging Face configs
mlx_config = {
    "n_mels": config.get("num_mel_bins", 128),
    "n_audio_state": config.get("d_model", 1280),
    "n_audio_ctx": config.get("max_source_positions", 1500),
    "n_audio_head": config.get("encoder_attention_heads", 20),
    "n_audio_layer": config.get("encoder_layers", 32),
    "n_vocab": config.get("vocab_size", 51866),
    "n_text_state": config.get("d_model", 1280),
    "n_text_ctx": config.get("max_target_positions", 448),
    "n_text_head": config.get("decoder_attention_heads", 20),
    "n_text_layer": config.get("decoder_layers", 4)
}

# Write the purely MLX-compatible config back
with open(path, 'w') as f:
    json.dump(mlx_config, f, indent=2)

print("Config mapped to native MLX parameters successfully!")
