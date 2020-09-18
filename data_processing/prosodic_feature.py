from pydub import AudioSegment
import parselmouth as pm
import numpy as np


def average(arr, n):
    """ Replace every "n" values by their average
    Args:
        arr: input array
        n:   number of elements to average on
    Returns:
        resulting array
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def derivative(x, f):
    """ Calculate numerical derivative (by FDM) of a 1d array
    Args:
        x: input space x
        f: Function of x
    Returns:
        der:  numerical derivative of f wrt x
    """

    x = 1000 * x  # from seconds to milliseconds

    # Normalization:
    dx = (x[1] - x[0])

    cf = np.convolve(f, [1, -1]) / dx

    # Remove unstable values
    der = cf[:-1].copy()
    der[0] = 0

    return der


def compute_prosody(audio_filename, time_step=0.05):
    audio = pm.Sound(audio_filename)

    # Extract pitch and intensity
    pitch = audio.to_pitch(time_step=time_step)
    intensity = audio.to_intensity(time_step=time_step)

    # Evenly spaced time steps
    times = np.arange(0, audio.get_total_duration() - time_step, time_step)

    # Compute prosodic features at each time step
    pitch_values = np.nan_to_num(
        np.asarray([pitch.get_value_at_time(t) for t in times]))
    intensity_values = np.nan_to_num(
        np.asarray([intensity.get_value(t) for t in times]))

    intensity_values = np.clip(
        intensity_values, np.finfo(intensity_values.dtype).eps, None)

    # Normalize features [Chiu '11]
    pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
    intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

    return pitch_norm, intensity_norm


def extract_prosodic_features(audio_filename):
    """
    Extract all 5 prosodic features
    Args:
        audio_filename:   file name for the audio to be used
    Returns:
        pros_feature:     energy, energy_der, pitch, pitch_der, pitch_ind
    """

    WINDOW_LENGTH = 5

    # Read audio from file
    sound = AudioSegment.from_file(audio_filename, format="wav")

    # Alternative prosodic features
    pitch, energy = compute_prosody(audio_filename, WINDOW_LENGTH / 1000)

    duration = len(sound) / 1000
    t = np.arange(0, duration, WINDOW_LENGTH / 1000)

    energy_der = derivative(t, energy)
    pitch_der = derivative(t, pitch)

    energy_sder = derivative(t, energy_der)
    pitch_sder = derivative(t, pitch_der)

    # Average everything in order to match the frequency
    energy = average(energy, 10)
    energy_der = average(energy_der, 10)
    energy_sder = average(energy_sder, 10)
    pitch = average(pitch, 10)
    pitch_der = average(pitch_der, 10)
    pitch_sder = average(pitch_sder, 10)

    # Cut them to the same size
    min_size = min(len(energy), len(energy_der), len(energy_sder),
                   len(pitch), len(pitch_der), len(pitch_sder))

    energy = energy[:min_size]
    energy_der = energy_der[:min_size]
    energy_sder = energy_sder[:min_size]
    pitch = pitch[:min_size]
    pitch_der = pitch_der[:min_size]
    pitch_sder = pitch_sder[:min_size]

    # Stack them all together
    # , pitch_ind))
    pros_feature = np.stack(
        (energy, energy_der, energy_sder, pitch, pitch_der, pitch_sder))

    # And reshape
    pros_feature = np.transpose(pros_feature)

    return pros_feature
