# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy
import librosa
import glob
import os

import matplotlib.pyplot as plt

from . import mel_features
from . import vggish_params

try:
  def aud_f_read(aud_file):
    aud_data, sr = librosa.load(aud_file)
    return aud_data, sr

except ImportError:

  def aud_f_read(aud_file):
    raise NotImplementedError('Audio file reading requires librosa package.')


def waveform_to_examples(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples


def aud_file_to_examples(aud_file):
  """Convenience wrapper around waveform_to_examples() for a common WAV or MP3 format.

  Args:
    aud_file: String path to a file, or a file-like object

  Returns:
    See waveform_to_examples.
  """
  aud_data, sr = aud_f_read(aud_file)
  #assert aud_data.dtype == np.int16, 'Bad sample type: %r' % aud_data.dtype
  #samples = aud_data / 32768.0  # Convert to [-1.0, +1.0]

  H_dir = "../data/H_audio"
  D_dir = "../data/D_audio"
  if os.path.basename(aud_file).split(".")[1][4:5] == 'H':
    aud_data_aug1 = same_class_augmentation(aud_data, H_dir)
  elif os.path.basename(aud_file).split(".")[1][4:5] == 'D':
    aud_data= add_noise1(aud_data)
    aud_data_aug1 = same_class_augmentation(aud_data, D_dir)
  aud_data_aug2 = pitch_shift_spectrogram(speed_numpy(aud_data))
  return waveform_to_examples(aud_data, sr), waveform_to_examples(aud_data_aug1, sr), waveform_to_examples(aud_data_aug2, sr),

def time_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)

def pitch_shift_spectrogram(spectrogram):
    """ 
    音高增强
    Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)

def add_noise1(x, w=0.004):
    # w：噪声因子
    output = x + w * np.random.normal(loc=0, scale=1, size=len(x))
    return output

def same_class_augmentation(wave, class_dir):
    """ Perform same class augmentation of the wave by loading a random segment
    from the class_dir and additively combine the wave with that segment.
    """
    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    aug_sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    (aug_sig, fs) = aud_f_read(aug_sig_path)
    alpha = np.random.rand()
    wave = (1.0-alpha)*wave + alpha*aug_sig
    return wave

def speed_numpy(samples, min_speed=0.9, max_speed=1.1):
    """
    线形插值速度增益
    :param samples: 音频数据，一维
    :param max_speed: 不能低于0.9，太低效果不好
    :param min_speed: 不能高于1.1，太高效果不好
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    speed = np.random.uniform(min_speed, max_speed)
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)
    return samples


if __name__=='__main__':
    x1,x2,x3 = aud_file_to_examples("../data/HD_audio/BaF1.1055H.1678278701.180827.1.1.wav")
    # print(x1)
    # print(x2)