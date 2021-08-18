#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

import unittest

sys.path.append(os.path.abspath(".."))

import speech2text

MODEL = "../models/output_graph.pbmm"
ALPHABET =  "../models/alphabet.txt"
LM = "../models/lm.binary"
TRIE = "../models/trie"

class TestSpeech2Text(unittest.TestCase):

    def setUp(self):
        self.audio_test_file1 = os.path.abspath("./data/open_the_door.wav")
        print (self.audio_test_file1)

    def test_detect_from_file(self):
        
        stt = speech2text.speech2text()
        stt.load_model(MODEL, ALPHABET, LM, TRIE)
        
        detected_text = stt.detect_file(self.audio_test_file1)
        self.assertEqual(detected_text, "open the door ")
        
if __name__ == '__main__':
    unittest.main()