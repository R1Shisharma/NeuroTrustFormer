
# ============================================================
# CHB-MIT EEG PREPROCESSING MODULE
# NeuroTrustFormer — HISS Major Revision
# ============================================================

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import mne


PROJECT = Path(
    "/content/drive/MyDrive/NeuroTrustFormer_HISS"
)

DATA_DIR = PROJECT / "data" / "chbmit"
RESULTS_DIR = PROJECT / "results"


# ------------------------------------------------------------
# LOCKED EEG SPECIFICATION
# ------------------------------------------------------------

TARGET_CHANNELS = [
    "C3-P3",
    "C4-P4",
    "CZ-PZ",
    "F3-C3",
    "F4-C4",
    "F7-T7",
    "F8-T8",
    "FP1-F3",
    "FP1-F7",
    "FP2-F4",
    "FP2-F8",
    "FZ-CZ",
    "P3-O1",
    "P4-O2",
    "P7-O1",
    "P8-O2",
    "T7-P7",
]

TARGET_CHANNELS_NORMALIZED = [
    ch.upper()
    for ch in TARGET_CHANNELS
]

SAMPLING_RATE = 256

WINDOW_SECONDS = 10

STEP_SECONDS = 5

N_SAMPLES = (
    SAMPLING_RATE *
    WINDOW_SECONDS
)

STEP_SAMPLES = (
    SAMPLING_RATE *
    STEP_SECONDS
)

N_CHANNELS = len(
    TARGET_CHANNELS
)


# ------------------------------------------------------------
# PATH RESOLUTION
# ------------------------------------------------------------

def resolve_edf_path(recording):

    recording = Path(
        str(recording)
    )

    candidates = [
        DATA_DIR / recording.name,
        DATA_DIR / str(recording),
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"EDF recording not found: {recording}"
    )


# ------------------------------------------------------------
# CHANNEL NAME NORMALIZATION
# ------------------------------------------------------------

def normalize_channel_name(name):

    return (
        str(name)
        .strip()
        .upper()
    )


# ------------------------------------------------------------
# LOAD ONE EEG WINDOW
# ------------------------------------------------------------

def load_eeg_window(
    recording,
    start_sec,
    end_sec,
):

    # --------------------------------------------------------
    # Window duration check
    # --------------------------------------------------------

    if abs(
        (
            float(end_sec)
            -
            float(start_sec)
        )
        -
        WINDOW_SECONDS
    ) > 1e-6:

        raise ValueError(
            "Window duration is not "
            "10 seconds: "
            f"{start_sec} -> {end_sec}"
        )

    # --------------------------------------------------------
    # Locate EDF
    # --------------------------------------------------------

    edf_path = resolve_edf_path(
        recording
    )

    # --------------------------------------------------------
    # Open EDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=False,
        verbose="ERROR"
    )

    # --------------------------------------------------------
    # Sampling-rate verification
    # --------------------------------------------------------

    sfreq = float(
        raw.info["sfreq"]
    )

    if not np.isclose(
        sfreq,
        SAMPLING_RATE,
        atol=1e-6
    ):

        del raw

        raise ValueError(
            f"Unexpected sampling rate "
            f"in {recording}: {sfreq}"
        )

    # --------------------------------------------------------
    # Build channel lookup
    # --------------------------------------------------------

    channel_map = {}

    for index, channel in enumerate(
        raw.ch_names
    ):

        normalized = (
            normalize_channel_name(
                channel
            )
        )

        if (
            normalized
            in TARGET_CHANNELS_NORMALIZED
        ):

            # Duplicate labels are NOT
            # arbitrarily selected.
            if normalized not in channel_map:

                channel_map[
                    normalized
                ] = index

            else:

                channel_map[
                    normalized
                ] = None

    # --------------------------------------------------------
    # Fixed tensor
    # --------------------------------------------------------

    eeg = np.zeros(
        (
            N_CHANNELS,
            N_SAMPLES
        ),
        dtype=np.float32
    )

    availability_mask = np.zeros(
        N_CHANNELS,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Window sample boundaries
    # --------------------------------------------------------

    start_sample = int(
        round(
            float(start_sec)
            *
            SAMPLING_RATE
        )
    )

    stop_sample = (
        start_sample
        +
        N_SAMPLES
    )

    # --------------------------------------------------------
    # Extract channels
    # --------------------------------------------------------

    for target_index, target in enumerate(
        TARGET_CHANNELS_NORMALIZED
    ):

        source_index = channel_map.get(
            target,
            None
        )

        # Missing or ambiguous channel.
        # Keep zero and mask=0.
        if source_index is None:
            continue

        signal = raw.get_data(
            picks=[source_index],
            start=start_sample,
            stop=stop_sample
        )[0]

        if len(signal) != N_SAMPLES:

            del raw

            raise ValueError(
                f"Unexpected sample count "
                f"for {recording}, "
                f"{target}: "
                f"{len(signal)}"
            )

        eeg[target_index] = (
            signal.astype(
                np.float32
            )
        )

        availability_mask[
            target_index
        ] = 1.0

    del raw

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    if eeg.shape != (
        N_CHANNELS,
        N_SAMPLES
    ):

        raise ValueError(
            f"Invalid EEG shape: "
            f"{eeg.shape}"
        )

    if availability_mask.shape != (
        N_CHANNELS,
    ):

        raise ValueError(
            "Invalid availability-mask "
            f"shape: "
            f"{availability_mask.shape}"
        )

    return (
        eeg,
        availability_mask
    )


# ------------------------------------------------------------
# TRAINING-ONLY NORMALIZATION
# ------------------------------------------------------------

class EEGNormalizer:

    def __init__(self):

        self.mean_ = None
        self.std_ = None
        self.fitted = False


    def fit(
        self,
        eeg_windows
    ):

        eeg_windows = np.asarray(
            eeg_windows,
            dtype=np.float64
        )

        if eeg_windows.ndim != 3:

            raise ValueError(
                "Expected EEG shape "
                "[N, channels, samples]"
            )

        if (
            eeg_windows.shape[1]
            !=
            N_CHANNELS
        ):

            raise ValueError(
                "Unexpected channel dimension."
            )

        flattened = (
            eeg_windows
            .transpose(
                1,
                0,
                2
            )
            .reshape(
                N_CHANNELS,
                -1
            )
        )

        self.mean_ = np.mean(
            flattened,
            axis=1
        )

        self.std_ = np.std(
            flattened,
            axis=1
        )

        self.std_[
            self.std_ < 1e-8
        ] = 1.0

        self.fitted = True

        return self


    def transform(
        self,
        eeg,
        availability_mask
    ):

        if not self.fitted:

            raise RuntimeError(
                "EEGNormalizer must be "
                "fitted on training data "
                "before transform()."
            )

        eeg = np.asarray(
            eeg,
            dtype=np.float32
        ).copy()

        mask = np.asarray(
            availability_mask,
            dtype=np.float32
        )

        for channel_index in range(
            N_CHANNELS
        ):

            if mask[channel_index] > 0:

                eeg[
                    channel_index
                ] = (
                    eeg[
                        channel_index
                    ]
                    -
                    self.mean_[
                        channel_index
                    ]
                ) / self.std_[
                    channel_index
                ]

            else:

                eeg[
                    channel_index
                ] = 0.0

        return eeg


# ------------------------------------------------------------
# PYTORCH DATASET
# ------------------------------------------------------------

class CHBMITEEGDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        normalizer=None,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.normalizer = normalizer


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        eeg, mask = load_eeg_window(
            recording=row[
                "recording"
            ],
            start_sec=row[
                "start_sec"
            ],
            end_sec=row[
                "end_sec"
            ],
        )

        if self.normalizer is not None:

            eeg = (
                self.normalizer.transform(
                    eeg,
                    mask
                )
            )

        return {
            "eeg": torch.from_numpy(
                eeg
            ),

            "channel_mask": torch.from_numpy(
                mask
            ),

            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long
            ),

            "case": str(
                row["case"]
            ),

            "recording": str(
                row["recording"]
            ),

            "window_index": int(
                row["window_index"]
            ),
        }
