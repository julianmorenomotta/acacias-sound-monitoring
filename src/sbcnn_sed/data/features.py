import torch
import torchaudio
from torchaudio import transforms


class MelSpectogramExtractor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 680,
        n_mels: int = 64,
        sequence_time: float = 1.0,
        sequence_hop_time: float = 0.5,
    ):

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels

        self.mel_spectogram_transform = transforms.MelSpectogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            center=False,
        )

        self.db_transform = transforms.AmplitudeToDB(stype="power", top_db=80)

        self.sequence_frames = int(sequence_time * sample_rate / hop_length)
        self.sequence_hop_frames = int(sequence_hop_time * sample_rate / hop_length)

    def extract(self, audio_path: str):
        waveform, orig_sr = torchaudio.load(audio_path)

        # resample
        if orig_sr != self.sample_rate:
            resampler = transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Full spectogram calculation
        mel_spec = self.mel_spectogram_transform(waveform)
        db_mel_spec = self.db_transform(mel_spec)

        sb_mel_spec = db_mel_spec.squeeze(0).T

        sequences = sb_mel_spec.unfold(
            dimension=0, size=self.sequence_frames, step=self.sequence_hop_frames
        )

        sequences = sequences.permute(0, 2, 1)

        return sequences
