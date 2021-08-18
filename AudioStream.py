#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import sys
import queue
import time
import traceback
import argparse
import wave
import os
import logging
import webrtcvad

import collections
import pyaudio

import numpy as np

from deepspeech import Model
from timeit import default_timer as timer


sys.path.append(os.path.abspath("."))
import speech2text

from audio_defaults import *

# max number of chunks(frames) written to data queue
# 0 for unlimited
DATA_QUEUE_MAX_SIZE = 0

DEBUG = True

logging.basicConfig(level=logging.INFO)

class AudioStream(object):
    """
        class to manage audio stream from input device.
        Note that the callback function will be called by in a separated thread
        Do not use bocking operations in callback. including stream read/write
        
    """
    def __init__(self, rate=RATE, chunk=CHUNK, callback=None):
        self._rate = rate
        self._chunk = chunk
        
        # data_queue that will be used in callback to collect audio chunks
        self.data_q = queue.Queue(DATA_QUEUE_MAX_SIZE)

        # instantiating a input/output stream to audio device
        self._paudio = pyaudio.PyAudio()
        self.closed = True
        if(callback is None):
            self._callback = self.queuing_callback

    def get_rate(self):
        return self._rate
    
    def get_chunk(self):
        return self._chunk
    
    def __enter__(self):
        self.stream = self._paudio.open(format = FORMAT,
                                        channels=CHANNELS,
                                        rate=self._rate,
                                        input=True,
                                        frames_per_buffer = self._chunk,
                                        stream_callback = self._callback)
        
        self.closed = False
        return self    
    
    def __exit__(self, exc_type, exc_value, tb):
                
#         if exc_type is not None:
#             traceback.print_exception(exc_type, exc_value, tb)
        self.stream.stop_stream()
        self.stream.close()
        self.closed = True
        self._paudio.terminate()
    
        
    
    def start_stream(self):
        self.stream.start_stream()
        self.closed = False

    def is_active(self):
        return self.stream.is_active()
        
    def stop_stream(self):
        self.stream.stop_stream()
        self.closed = True

    def queuing_callback(self, in_data, frame_count, time_info, status_flags):
        """
        this is queing callback that will run on a separated thread.
        Do not use read/write or any blocking operation in this call. 
        """ 
#         logging.debug("frame_count:{} time_info:{}, status_flags:{}".
#                     format(frame_count, time_info, status_flags))
        self.data_q.put(in_data)
        return None, pyaudio.paContinue
    
    def get_frame(self):
        if not self.closed:
            try:
                frames = self.data_q.get(timeout=0.3)
                return frames
            except queue.Empty:
                pass


class VadBuffer (object):
    """ 
    This is special deque implementation with counter for audio frame properties 
    to calculate voice frames with high performance. 
    This is NOT a thread safe deque implementation
    """
    
    def __init__(self, maxlen):
        self.buff = collections.deque(maxlen = maxlen)
        self.__num_voice = 0
        self.__size = 0

    def append(self, frame, is_speech):
        if self.__size == self.buff.maxlen:
            # most left frame in queue is speech.
            f, s = self.buff.popleft()
            self.__size -= 1
            
            if(s == True):
                self.__num_voice -= 1
        
        self.buff.append((frame, is_speech))
        
        self.__num_voice += 1 if is_speech else 0
        self.__size += 1

    def clear(self):
        self.__num_voice = 0
        self.__size  = 0
        self.buff.clear()
        
    @property
    def num_voice(self):
        return self.__num_voice
    
    @property
    def size(self):
        return self.__size

    @property
    def num_non_voice(self):
        return self.__size - self.__num_voice
    
    @property
    def voice_frame_ratio(self):
        return self.num_voice/self.buff.maxlen
    
    @property
    def non_voice_frame_ratio(self):
        return self.num_non_voice/self.buff.maxlen
    
    def get_data_list(self):
        return [f[0] for f in self.buff]
 
    
class VadFilter(object):

    def __init__(self, audio_stream, mode, vad = None):
        """
        Live Audio Stream with buffering and vad filter
        """
        if(mode not in [0,1,2,3]):
            raise(ValueError("invalid mode:{}, should be [0,1,2,3]".format(mode)))

        if(vad != None):
            self.vad  = vad 
        else: 
            self.vad = webrtcvad.Vad(mode)
        
        self.audio_stream = audio_stream


    def audio_frame_generator(self):
        while self.audio_stream.is_active():
            frame = self.audio_stream.get_frame()
            if(frame == None):
                continue
            yield frame
            
    def voice_segment_collector(self, vad_buffer_ms):
        
        """
        Examines audio stream and collects voice frames.
        Frames yielded when speech offset is detected. 
        The detection mechanism uses vad_buffer length of vad_num_frames.
        
        Segment is considered speech when vad_buffer_ratio of total frames 
        in the vad_buffer are speech.
        
        The same logic is used for non-voice segments.
        When speech_onset is True the frames will be added to the voiced_frames.
        """
        # ratio of frames in the buffer to be considered voice ot not. 
        VAD_BUFFER_RATIO = 0.9
        
        rate = self.audio_stream.get_rate()
        chunk = self.audio_stream.get_chunk()
        frame_ms = (1/rate ) * chunk * 1000
        
        # number of frames in the VadBuffer
        vad_num_frames = int(vad_buffer_ms / frame_ms)

        logging.info("frame_ms :{} ms".format(frame_ms))
        logging.info("vad buffer len:{}".format(vad_num_frames))
        
        vad_buffer = VadBuffer(maxlen=vad_num_frames)
        
        speech_onset = False
        voiced_frames = []
        dbg_speech_frames = ''
        
        frames = self.audio_frame_generator()
        for frame in frames:
            
            is_speech = self.vad.is_speech(frame, rate)

            # this is just for visual debugging 
            if DEBUG:
                dbg_speech_frames  += '1' if is_speech else '0'

            if not speech_onset:
                vad_buffer.append(frame, is_speech)
                
                if(vad_buffer.voice_frame_ratio > VAD_BUFFER_RATIO):
                    speech_onset = True
                    voiced_frames.extend(vad_buffer.get_data_list())
                    vad_buffer.clear()
            else:
                voiced_frames.append(frame)
                
                vad_buffer.append(frame, is_speech)                
                if vad_buffer.non_voice_frame_ratio > VAD_BUFFER_RATIO:
                    speech_onset = False

                    if DEBUG:
                        logging.info(dbg_speech_frames) 
                        dbg_speech_frames  = ''
                        
                    yield b''.join([f for f in voiced_frames])                    
#                     yield np.array(voiced_frames)
                    vad_buffer.clear()
                    voiced_frames = []
        if DEBUG:
            logging.info(dbg_speech_frames)
            dbg_speech_frames = ''
        
        if voiced_frames:
            print ("voiced_frames not empty")
            yield b''.join([f for f in voiced_frames])
#             yield np.array(voiced_frames)
    
def main (model, alphabet, lm, trie):

    # loading the model as it takes time to do so.
    stt = speech2text.speech2text()
    stt.load_model(model, alphabet, lm, trie)
    
    with AudioStream(RATE, CHUNK) as audio_stream:
        print("recording started...")
        vad_filter = VadFilter(audio_stream, DETECTION_MODE)
        segments = vad_filter.voice_segment_collector(200)
    
        for segment in segments:
            print('Recognizing Speech...')
            speech_text = stt.detect_buffer(segment)
            print(speech_text)
            
            # just for fun. say this to exit
            if(speech_text.strip() == "finished"):
                audio_stream.stop_stream()
                time.sleep(0.3)
                sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--alphabet', required=True,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('--lm', nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('--trie', nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')

    parser.add_argument('--audio', required=False,
                        help='Path to the audio file to run (WAV format)')

    args = parser.parse_args()

    main(args.model, args.alphabet, args.lm, args.trie)
