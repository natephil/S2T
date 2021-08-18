#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import webrtcvad
import unittest
import unittest.mock as mock


sys.path.append(os.path.abspath(".."))

import AudioStream
import audio_defaults


MODEL = "../models/output_graph.pbmm"
ALPHABET =  "../models/alphabet.txt"
LM = "../models/lm.binary"
TRIE = "../models/trie"

class TestAudioStream(unittest.TestCase):    


    def test_stream_getters(self):
        with AudioStream.AudioStream(audio_defaults.RATE, audio_defaults.CHUNK) as stream:
            self.assertEqual(stream.get_rate(), audio_defaults.RATE)
            self.assertEqual(stream.get_chunk(), audio_defaults.CHUNK)
            
    def test_stream_status(self):
        with AudioStream.AudioStream(audio_defaults.RATE, audio_defaults.CHUNK) as stream:
            self.assertEqual(stream.is_active(), True)
            
            stream.stop_stream()
            self.assertEqual(stream.is_active(), False)
            
            stream.start_stream()
            self.assertEqual(stream.is_active(), True)
            
    def test_stream_queue(self):

        with AudioStream.AudioStream(audio_defaults.RATE, audio_defaults.CHUNK) as stream:
            frame = stream.get_frame()            
            counter = 0
            while counter<10:
                self.assertEqual(len(frame), audio_defaults.CHUNK * 2)
                counter +=1

class TestVadFilter(unittest.TestCase):

    def test_init_pass(self):
        vf = None
        with AudioStream.AudioStream(audio_defaults.RATE, audio_defaults.CHUNK) as stream:
            vf = AudioStream.VadFilter(stream, 1)
            self.assertNotEqual(vf, None)
            self.assertIsInstance(vf, AudioStream.VadFilter)
            
    def test_init_fails(self):
        with AudioStream.AudioStream(audio_defaults.RATE, audio_defaults.CHUNK) as stream:
            self.assertRaises(ValueError, AudioStream.VadFilter, stream, 4)
        
    def test_audio_generator(self):
        """
        this test is more of integration test than unit-test. 
        we are testing reading from Microphone and being able to 
        generate frames from pyAudio buffer it just .
        """
        counter = 0
        with AudioStream.AudioStream(audio_defaults.RATE, audio_defaults.CHUNK) as stream:
            vf = AudioStream.VadFilter(stream, 1)
            
            frames = vf.audio_frame_generator()
            for frame in frames:
                self.assertEqual(len(frame), audio_defaults.CHUNK * 2)
                counter += 1
                if(counter == 10):
                    break
    
    def test_mock_is_speech_method(self):
        """
        this is just a test to demonstrate mocking of Vad on import 
        """
        with mock.patch('webrtcvad.Vad', ) as MockVadObj:
            vad_return_items = [0,1,0,1,0,1]
            MockVadObj.return_value.is_speech.side_effect = vad_return_items
            mvo = MockVadObj(3)
            for item in vad_return_items:
                self.assertEqual(mvo.is_speech(), item)

    
    def test_mock_audio_stream_get_frame(self):
        """
        this is to test mocking of only a methond of a class
        using mock.patch.object we mock get_frame of the class and create MockAudioStream.
        This helps defining return items of that method. 
        """
        with mock.patch.object(AudioStream.AudioStream, 'get_frame') as MockAudioStream:
            mas_return_items = [b'data1', b'data2', b'data3', b'data4', b'data5',
                                b'data6', b'data7', b'data8', b'data9', b'data10']
            MockAudioStream.return_value.get_frame.side_effect = mas_return_items
            mas= MockAudioStream()
            for item in mas_return_items:
                mock_frame = mas.get_frame()
                self.assertEqual(mock_frame , item)
#                 print ("mock_frame:{}".format(mock_frame))       
    
    
    def test_audio_generator_with_mock_audio_stream(self):
        """
        testing VadFilter.audio_frame_generator using mocked Audio Stream.
        
        """
        with mock.patch.object(AudioStream.AudioStream, 'get_frame') as MockAudioStream:
            mas_return_items = [b'data1', b'data2', b'data3', b'data4', b'data5',
                                b'data6', b'data7', b'data8', b'data9', b'data10']
            MockAudioStream.return_value.get_frame.side_effect = mas_return_items
            mas= MockAudioStream()
            
            vf = AudioStream.VadFilter(audio_stream = mas, mode = 3 )
            frames = vf.audio_frame_generator()
            for item in mas_return_items:
                frame = next(frames)
                self.assertEqual(frame ,item)

    def test_voice_segment_collector(self):
        """
        VAD segment collection test.
        In this test we make sure that the audiofile is segmented when 
        in between voiced audio section.     
        """
        with mock.patch('webrtcvad.Vad', ) as MockVadObj:
            vad_return_items = [0,0,1,1,1,1,1,1,0,0]
            MockVadObj.return_value.is_speech.side_effect = vad_return_items

            with mock.patch.object(AudioStream.AudioStream, 'get_frame') as MockAudioStream:
                mas_return_items = [b'data1', b'data2', b'data3', b'data4', b'data5',
                                    b'data6', b'data7', b'data8', b'data9', b'data10']
                MockAudioStream.return_value.get_frame.side_effect = mas_return_items
                mas= MockAudioStream()
                
                vf = AudioStream.VadFilter(audio_stream = mas, mode = 3)
                segments = vf.voice_segment_collector(100)
                segment = next(segments)
#                 print(segment)
                self.assertEqual(segment, b'data3data4data5data6data7data8data9')
    
    def mock_VAD_segment_collector(self, mock_vad_return, 
                                   mock_audio_frames, vad_buffer_ms):
        """
        This is a test mock helper class that will be used for different 
        frame audio, and speech detection scenarios
        """
        with mock.patch('webrtcvad.Vad', ) as MockVadObj:
             MockVadObj.return_value.is_speech.side_effect = mock_vad_return   
             with mock.patch.object(AudioStream.AudioStream, 'get_frame') as MockAudioStream:
                MockAudioStream.return_value.get_frame.side_effect = mock_audio_frames
                mas= MockAudioStream()
                
                vf = AudioStream.VadFilter(audio_stream = mas, mode = 3)
                segments = vf.voice_segment_collector(vad_buffer_ms)
                ret_segments = []
                while True:
                    try:
                        segment = next(segments)
                        ret_segments.append(segment)
                    except Exception as e:
                        print ("Expected caught exception: {} ok".format(e))
                        break
                    
                return ret_segments

    def test_VAD_segment_collector_single_segment(self):
        vad_detection_items = [0,0,1,1,1,1,1,1,0,0]
        mocked_audio_frames = [b'data1', b'data2', b'data3', b'data4', b'data5',
                            b'data6', b'data7', b'data8', b'data9', b'data10']
        segments = self.mock_VAD_segment_collector(vad_detection_items,
                                                   mocked_audio_frames, 
                                                   200)
        # only one segment in the audio stream
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], b'data3data4data5data6data7data8data9')
        
    def test_VAD_segment_collector_multiple_segment(self):
        vad_detection_items = [0,0,1,1,1,1,1,1,0,0,
                               0,1,1,1,0,0,0,0,0,0]
        mocked_audio_frames = [b'data1', b'data2', b'data3', b'data4', b'data5',
                               b'data6', b'data7', b'data8', b'data9', b'data10', 
                               b'data11', b'data12', b'data13', b'data14', b'data15',
                               b'data16', b'data17', b'data18', b'data19', b'data20']
        
        segments = self.mock_VAD_segment_collector(vad_detection_items, 
                                                   mocked_audio_frames, 
                                                   200)
        # making sure multiple segments of Audio is detected
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], b'data3data4data5data6data7data8data9')
        self.assertEqual(segments[1], b'data12data13data14data15')
        print (segments)

    def test_VAD_segment_collector_no_speech(self):
        vad_detection_items = [0,0,0,0,0,0,0,0,0,0]
        mocked_audio_frames = [b'data1', b'data2', b'data3', b'data4', b'data5',
                               b'data6', b'data7', b'data8', b'data9', b'data10'] 
        
        segments = self.mock_VAD_segment_collector(vad_detection_items, 
                                                   mocked_audio_frames, 
                                                   200)
        # making sure no segment is detected
        self.assertEqual(len(segments), 0)

    # this might be because the way test or software.     
    @unittest.skip("the last segment not being detected, possible bug!")
    def test_VAD_segment_collector_trailing_segment(self):
        
        vad_detection_items = [0,0,1,1,1,1,1,1,0,0,
                               0,0,0,0,0,1,1,1,1,1]
        mocked_audio_frames = [b'data1', b'data2', b'data3', b'data4', b'data5',
                               b'data6', b'data7', b'data8', b'data9', b'data10', 
                               b'data11', b'data12', b'data13', b'data14', b'data15',
                               b'data16', b'data17', b'data18', b'data19', b'data20',
                               b'data16', b'data17', b'data18', b'data19', b'data20']
        
        segments = self.mock_VAD_segment_collector(vad_detection_items, 
                                                   mocked_audio_frames, 
                                                   200)
        
        # making sure multiple segments of Audio is detected. 
        # we had some issues with last segment.
        self.assertEqual(len(segments), 2)
        
        self.assertEqual(segments[0], b'data3data4data5data6data7data8data9')
        self.assertEqual(segments[1], b'data12data13data14data15')
        print (segments)

    @unittest.skip("is this a bug??")
    def test_VAD_segment_collector_all_speech(self):
        vad_detection_items = [1,1,1,1,1,1,1,1,1]
        mocked_audio_frames = [b'data1', b'data2', b'data3', b'data4', b'data5',
                               b'data6', b'data7', b'data8', b'data9', b'data10'] 
        
        segments = self.mock_VAD_segment_collector(vad_detection_items, mocked_audio_frames, 200)
        # making sure no segment is detected
        self.assertEqual(len(segments), 1)
        
class TestVadBuffer(unittest.TestCase):
    
    def setUp(self):
        self.vb = AudioStream.VadBuffer(5)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.vb.size ,0)
        self.assertEqual(self.vb.num_voice ,0)
        
        
    def test_append(self):
        frame = b'some meaningless frame data1'
    
        self.vb.append(frame, True)
        self.assertEqual(self.vb.num_voice,1)
        self.assertEqual(self.vb.num_non_voice,0)
        self.assertEqual( self.vb.size , 1)

        self.vb.append(frame, True)
        self.assertEqual(self.vb.num_voice,2)
        self.assertEqual(self.vb.num_non_voice,0)
        self.assertEqual( self.vb.size , 2)

        self.vb.append(frame, False)
        self.vb.append(frame, False)        
        self.vb.append(frame, True)
        # [1,1,0,0,1]
        
        self.assertEqual(self.vb.num_voice,3)
        self.assertEqual(self.vb.num_non_voice,2)
        self.assertEqual( self.vb.size , 5)

        self.assertEqual(self.vb.voice_frame_ratio, 0.6)
        self.assertEqual(self.vb.non_voice_frame_ratio, 0.4)

        
        # should not go beyond max len and first frame should be removed        
        self.vb.append(frame, False)
        # [1,0,0,1,0]
        
        self.assertEqual(self.vb.num_voice,2)
        self.assertEqual(self.vb.num_non_voice,3)
        self.assertEqual( self.vb.size , 5)
 
        self.assertEqual(self.vb.voice_frame_ratio, 0.4)
        self.assertEqual(self.vb.non_voice_frame_ratio, 0.6)
        
        data_list  = self.vb.get_data_list()
        self.assertEqual(data_list[0], frame)
        self.assertEqual(data_list[4], frame)
        self.test_clear()
    
        
#     def test_voice_frame_ratio(self):
#          self.assertRaises(AttributeError, self.vb.voice_frame_ratio)
# #         self.assertEqual(self.vb.voice_frame_ratio, 0.4)
        
    def test_clear(self):
        self.vb.clear()
        self.assertEqual(self.vb.num_voice,0)
        self.assertEqual(self.vb.num_non_voice,0)
        self.assertEqual( self.vb.size , 0)

        
        
if __name__ == '__main__':
    unittest.main()































