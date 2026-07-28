import torch
import numpy as np
import librosa
import soundfile as sf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MelSpectrogramExtractor:
    """
    Extract lo-scaled Mel-spectogram sequences.
    This is a re-implementation of the D-CASE models
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 680,
        n_mels: int = 64,
        sequence_time: float = 1.0,
        sequence_hop_time: float = 0.5,
        pad_mode: str = "reflect",
    ):

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.pad_mode = pad_mode
        self.sequence_time = sequence_time
        self.sequence_hop_time = sequence_hop_time

        self.sequence_frames = int(
            librosa.core.time_to_frames(
                sequence_time, sr=sample_rate, hop_length=hop_length
            )
        )
        self.sequence_hop = int(
            librosa.core.time_to_frames(
                sequence_hop_time, sr=sample_rate, hop_length=hop_length
            )
        )

        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, htk=True, fmax=None
        )

    def load_audio(self, filename: str | Path, mono: bool = True) -> np.ndarray:
        """
        Loads an audio signal, converts to mono, and resamples to self.sample_rate
        """
        audio, orig_sr = sf.read(str(filename))

        if len(audio.shape) > 1 and mono:
            audio = audio[:, 0]

        audio = np.asfortranarray(audio)

        if orig_sr != self.sample_rate:
            logger.info(f"r-esampling from {orig_sr} to {self.sample_rate}")
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

        return audio

    def _pad_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Pading audio
        """
        if self.sequence_time > 0 and self.pad_mode is not None:
            sequence_samples = self.sequence_frames * self.hop_length + self.win_length

            if len(audio) < sequence_samples:
                audio = librosa.util.fix_length(
                    audio, size=sequence_samples, axis=0, mode=self.pad_mode
                )
            else:
                if self.sequence_hop_time > 0:
                    audio_frames = int(
                        (len(audio) - self.win_length) / self.hop_length
                    ) + int(((len(audio) - self.win_length) % self.hop_length) > 0)
                    n_sequences = int(
                        (audio_frames - self.sequence_frames) / self.sequence_hop
                    ) + int(
                        ((audio_frames - self.sequence_frames) % self.sequence_hop) > 0
                    )
                    new_frames = n_sequences * self.sequence_hop + self.sequence_frames
                    new_samples = new_frames * self.hop_length + self.win_length
                    audio = librosa.util.fix_length(
                        audio, size=new_samples, axis=0, mode=self.pad_mode
                    )
                else:
                    audio = audio[:sequence_samples]

        return audio

    def _convert_to_sequences(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Sliding window conversion.
        """
        if self.sequence_time > 0 and self.sequence_hop_time > 0:
            mel_spec_contigous = np.ascontiguousarray(mel_spec)
            frames = librosa.util.frame(
                mel_spec_contigous,
                frame_length=self.sequence_frames,
                hop_length=self.sequence_hop,
                axis=0,
            )
            return frames
        else:
            return np.expand_dims(mel_spec, axis=0)[:, : self.sequence_frames]

    def extract(
        self, audio_input: str | Path | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the mel spectograms for raw audio signal.
        """
        if isinstance(audio_input, (str, Path)):
            audio = self.load_audio(audio_input)
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.numpy().squeeze()
        else:
            audio = audio_input

        audio_padded = self._pad_audio(audio)

        stft = librosa.core.stft(
            audio_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
        )

        power_spec = np.abs(stft) ** 2
        mel_spec = self.mel_basis.dot(power_spec)
        mel_spec_db = librosa.power_to_db(mel_spec)
        mel_spec_db_t = mel_spec_db.T
        sequences = self._convert_to_sequences(mel_spec_db_t)

        return torch.tensor(sequences, dtype=torch.float32)

    def __call__(self, audio: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy().squeeze()
        return self.extract(audio)
