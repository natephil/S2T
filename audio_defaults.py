# -*- coding: utf-8 -*-
import pyaudio

###################
# pyAudio settings 
####################

# frames per bufferr
CHUNK = 480

# audio sample number of bytes
# this also refer as width or format
FORMAT = pyaudio.paInt16

# this has to be mono for voice detection
# mono/stereo or lef/right
CHANNELS = 1

# audio sampling rate
RATE = 16000

###################
# VAD settings 
####################
# webrtc.Vad aggressiveness level 
DETECTION_MODE = 3

####################
# DeepSpeech settings
####################
# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.50

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 2.10

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9