from semanticodec import SemantiCodec
import soundfile as sf

semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=16384)  # 1.35 kbps
filepath = "test.wav"

tokens = semanticodec.encode(filepath)
waveform = semanticodec.decode(tokens)

sf.write("test_output_100.wav", waveform[0, 0], 16000)
#########################################################

semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=16384)  # 0.68 kbps

tokens = semanticodec.encode(filepath)
waveform = semanticodec.decode(tokens)

sf.write("test_output_50.wav", waveform[0, 0], 16000)
#########################################################

semanticodec = SemantiCodec(token_rate=25, semantic_vocab_size=16384)  # 0.34 kbps

tokens = semanticodec.encode(filepath)
waveform = semanticodec.decode(tokens)

sf.write("test_output_25.wav", waveform[0, 0], 16000)
