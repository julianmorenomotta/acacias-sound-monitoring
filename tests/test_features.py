import torch
import numpy as np
from sbcnn_sed.data.features import MelSpectrogramExtractor


class TestMelSpectrogramExtractor:
    """Unit tests for MelSpectrogramExtractor."""

    def test_default_constructor(self):
        """
        GIVEN default parameters
        THEN sequence_frames=32, sequence_hop=16, n_mels=64

        sequence_frames = time_to_frames(1.0s, 22050 Hz, hop=680) = 32
        sequence_hop    = time_to_frames(0.5s, 22050 Hz, hop=680) = 16
        These values must match the model's expected input shape (32, 64).
        """
        extractor = MelSpectrogramExtractor()
        # 1.0s at 22050 Hz with hop_length=680 => 32 frames
        assert extractor.sequence_frames == 32
        # 0.5s hop at 22050 Hz with hop_length=680 => 16 frames
        assert extractor.sequence_hop == 16
        assert extractor.n_mels == 64

    def test_extract_numpy_1s_returns_one_sequence(self):
        """
        GIVEN a 1-second sine wave as a numpy array
        WHEN extract() is called
        THEN the output is a float32 tensor of shape (1, 32, 64)

        np.linspace(0, 1.0, sr, endpoint=False) creates exactly sr samples
        over 1 second without duplicating the endpoint. Combined with
        np.sin(2 * pi * 440 * t) this generates a clean 440 Hz tone (note A4).
        """
        extractor = MelSpectrogramExtractor()
        sr = 22050  # must match the default sample rate
        # 22050 evenly spaced samples over 1 second (no duplicate endpoint)
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # 440 Hz sine wave (musical note A4) — any clean tone works
        audio = np.sin(2 * np.pi * 440 * t)
        result = extractor.extract(audio)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        # (1 sequence, 32 time frames, 64 mel bands)
        assert result.shape == (1, 32, 64)

    def test_extract_torch_tensor_same_as_numpy(self):
        """
        GIVEN the same audio as a torch tensor and a numpy array
        WHEN extract() is called on both
        THEN the results are identical

        The extract() method has separate code paths for numpy arrays
        (else branch) and torch tensors (elif isinstance branch).
        Both should produce identical output.
        """
        extractor = MelSpectrogramExtractor()
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)
        # numpy code path (hits 'else' branch in extract())
        result_np = extractor.extract(audio)
        # torch code path (hits 'elif isinstance(audio_input, torch.Tensor)' branch)
        result_torch = extractor.extract(torch.from_numpy(audio))
        assert result_torch.shape == result_np.shape
        assert torch.allclose(result_torch, result_np)

    def test_long_audio_produces_multiple_sequences(self):
        """
        GIVEN a 2-second sine wave
        WHEN extract() is called
        THEN the output has multiple sequences, each of shape (32, 64)

        The sliding window (frame_length=32, hop_length=16) over 2 seconds
        of audio should produce more than 1 overlapping sequence.
        """
        extractor = MelSpectrogramExtractor()
        sr = 22050
        # 2 seconds of audio at 22050 Hz => 44100 samples
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)
        result = extractor.extract(audio)
        assert result.dim() == 3
        assert result.shape[0] > 1  # multiple sequences
        assert result.shape[1] == 32  # each is 32 frames
        assert result.shape[2] == 64  # each is 64 mel bands

    def test_short_audio_gets_padded(self):
        """
        GIVEN a very short audio clip (0.1s)
        WHEN extract() is called
        THEN the audio is padded to produce at least 1 valid sequence

        sr // 10 gives 2205 samples for 0.1s at 22050 Hz.
        This is shorter than the minimum sequence length, so
        _pad_audio() should pad it to exactly 1 sequence.
        """
        extractor = MelSpectrogramExtractor()
        sr = 22050
        # 0.1s at 22050 Hz => 2205 samples (sr // 10)
        t = np.linspace(0, 0.1, sr // 10, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)
        result = extractor.extract(audio)
        # padding should produce exactly 1 sequence
        assert result.shape == (1, 32, 64)
        # no NaN or Inf from padding (reflect padding adds mirrored values)
        assert torch.isfinite(result).all()

    def test_extract_returns_finite_values(self):
        """
        GIVEN clean synthetic audio
        WHEN extract() is called
        THEN all output values are finite (no NaN, no Inf)

        The mel spectrogram pipeline (STFT -> power -> mel -> dB -> sequence)
        should never produce NaN or Inf from clean audio input.
        """
        extractor = MelSpectrogramExtractor()
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)
        result = extractor.extract(audio)
        assert torch.isfinite(result).all()
