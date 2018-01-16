
from __future__ import absolute_import, division, print_function
from scipy.io.wavfile import read as scipy_read
import os
import pandas as pd
from deepspeech.model import Model

test = '/home/nasdin/Documents/github/Kaggle/2017Q4Tensorflow_Speech_Recognition/data/test/audio/'
files = {str(file):test+file for file in os.listdir(test) if file.endswith('.wav')}

output_graph = '/home/nasdin/Documents/github/DeepSpeech/models/output_graph.pb'
alphabet = '/home/nasdin/Documents/github/DeepSpeech/models/alphabet.txt'
lm_binary = '/home/nasdin/Documents/github/DeepSpeech/models/lm.binary'
trie = '/home/nasdin/Documents/github/DeepSpeech/models/trie'

prediction = pd.read_csv('data/sample_submission.csv')

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

ds = Model(lm_binary,N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
ds.enableDecoderWithLM(alphabet, lm_binary, trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

# #python to batch process
# import subprocess
#
# for index, i in enumerate(prediction.fname):
#     argument = output_graph +" "+ files[i]+" " + alphabet+" " + lm_binary+" " + trie
#
#
#     with subprocess.Popen(['deepspeech', argument], shell=True,stdout=subprocess.PIPE) as item:
#         for line in item.stdout:
#             print (line)


for index,i in enumerate(prediction.fname):
    with scipy_read(files[i]) as a:
        fs,audio = a
        #get the inference and replace it inside the pandas dataframe
        prediction.set_value(index,'label',ds.stt(audio,fs))

#save the prediction
prediction.to_csv("predictions.csv")