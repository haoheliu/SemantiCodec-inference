from semanticodec import SemantiCodec
import soundfile as sf

def test_semanticodec(token_rate, semantic_vocab_size, test_id):
    print(f"Testing with token_rate: {token_rate}, semantic_vocab_size: {semantic_vocab_size}")
    semanticodec = SemantiCodec(token_rate=token_rate, semantic_vocab_size=semantic_vocab_size)
    filepath = "test.wav"
    
    # Encoding and decoding process
    tokens = semanticodec.encode(filepath)
    waveform = semanticodec.decode(tokens)
    
    # Writing the output to a file
    output_filename = f"output_{test_id}.wav"
    sf.write(output_filename, waveform[0, 0], 16000)
    print(f"Output written to {output_filename}\n")
    del semanticodec

# Test cases
test_cases = [
    (100, 32768),
    (50, 32768),
    (25, 32768),
    (100, 16384),
    (50, 16384),
    (25, 16384),
    (100, 8192),
    (50, 8192),
    (25, 8192),
    (100, 4096),
    (50, 4096),
    (25, 4096)
]

# Running all test cases
for idx, (rate, vocab_size) in enumerate(test_cases, start=1):
    test_semanticodec(rate, vocab_size, idx)