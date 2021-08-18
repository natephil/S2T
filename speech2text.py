#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import wave

from deepspeech import Model
from timeit import default_timer as timer
import audio_defaults 

MODEL = "./models/output_graph.pbmm"
ALPHABET =  "./models/alphabet.txt"
LM = "./models/lm.binary"
TRIE = "./models/trie"


class speech2text(object):

    def __init__(self):
        self.ds = None
        
    def load_model(self, model, alphabet, lm, trie):
        print('Loading model from file {}'.format(model), file=sys.stderr)
        model_load_start = timer()
        self.ds = Model(model, audio_defaults.N_FEATURES, audio_defaults.N_CONTEXT, alphabet, audio_defaults.BEAM_WIDTH)
        model_load_end = timer() - model_load_start
        print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)
    
        if lm and trie:
            print('Loading language model from files {} {}'.format(lm, trie), file=sys.stderr)
            lm_load_start = timer()
            self.ds.enableDecoderWithLM(alphabet, lm, trie, audio_defaults.LM_WEIGHT,
                                   audio_defaults.VALID_WORD_COUNT_WEIGHT)
            lm_load_end = timer() - lm_load_start
            print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)
        else: 
            print('Empty Language Model or trie {} {}'.format(lm, trie), file=sys.stderr)
    
    
    def detect_buffer(self, audio_buffer):
        
        audio = np.frombuffer(audio_buffer, np.int16)
        
        inference_start = timer()
        speech_text = self.ds.stt(audio, audio_defaults.RATE)
        inference_end = timer() - inference_start
        
        audio_length = float(len(audio_buffer)/(audio_defaults.RATE * audio_defaults.CHANNELS * 2))
        print ("Inference took {0:.3f}s for {0:.3f}s audio buffer".
               format(inference_end, audio_length))

        return speech_text    
    
    def detect_file(self, file_name = None):
        
        speech_text = None
        
        with wave.open(file_name) as fin:
            fs = fin.getframerate()

            print ("filename:{} framerate:{}".format(file_name, fs))
            if fs != audio_defaults.RATE:
                print('Warning: original sample rate ({}) is different than 16kHz.\
                 Resampling might produce erratic speech recognition.'.
                 format(fs), file=sys.stderr)
            else:
                audio_buffer = fin.readframes(fin.getnframes())
                speech_text = self.detect_buffer(audio_buffer)
                
        return speech_text


