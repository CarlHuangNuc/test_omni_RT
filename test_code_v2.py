import cv2
import numpy as np
import requests
import json
import base64
import time
import threading
from queue import Queue
import queue
import tempfile
import os
import ffmpeg

import soundfile as sf
from openai import OpenAI
import pyaudio

client = OpenAI(
    api_key="sk-3e9cf6e6052a4c7dba9dd02bc30bd529",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
context_len = 10 # s video
class OpenAI4oStream:
    def __init__(self, api_key, max_frames=context_len):
        self.api_key = api_key
        self.max_frames = max_frames
        self.video_queue = Queue(maxsize=max_frames)
        self.audio_queue = Queue(maxsize=max_frames * 16)
        self.stop_event = threading.Event()
        

    def capture_video(self, camera_index=0):
        """从摄像头捕获视频帧"""
        frame_id=0
        frames_save = 30  ### 抽帧 1s 一帧
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError("无法打开摄像头")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("无法获取视频帧")
                break

            # 调整帧大小以减小数据量
            frame = cv2.resize(frame, (640, 480))
            frame_id = frame_id + 1
            # 将帧放入队列，如果队列满则丢弃最旧的帧

            if frame_id % frames_save == 0:
                print("frame_id====",frame_id/30)
                if self.video_queue.full():
                    print("video queue ....full,,remove the oldest frame")
                    self.video_queue.get_nowait()
                self.video_queue.put_nowait(frame)

            # 显示视频（可选）
            cv2.imshow('Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def capture_audio(self):
        """捕获音频流（需要安装 pyaudio）"""
        try:
            import pyaudio
        except ImportError:
            print("需要安装 pyaudio 库来捕获音频")
            return

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("****** 开始采集 video with audio ******")
        audio_frame_id = 0
        while not self.stop_event.is_set():
            data = stream.read(CHUNK)

            # 将音频数据放入队列
            audio_frame_id = audio_frame_id + 1
            if self.audio_queue.full():
                self.audio_queue.get_nowait()
            self.audio_queue.put_nowait(data)
        print("****** 停止录音 ******")

        stream.stop_stream()
        stream.close()
        p.terminate()


    def process_stream(self, prompt="描述这个场景"):
        """处理音视频流并发送到 OpenAI 4o API"""

        # 创建临时文件存储视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        # 创建临时文件存储音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name

        try:
            # 等待队列中有足够的数据
            while self.video_queue.qsize() < context_len or self.audio_queue.qsize() < context_len*16:
                time.sleep(1)

            # 从队列中获取帧并创建视频
            self._create_video_from_frames(video_path)
            # 从队列中获取音频数据并创建音频文件
            self._create_audio_from_chunks(audio_path)

            # 发送到 OpenAI API
            response = self._send_to_openai(video_path, audio_path, prompt)
            return response

        finally:
            # 清理临时文件
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
    def _create_video_from_frames(self, output_path):
        """从帧队列创建视频文件"""
        fps = 1.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (640, 480)

        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while not self.video_queue.empty():
            frame = self.video_queue.get_nowait()
            out.write(frame)

        out.release()

    def _create_audio_from_chunks(self, output_path):
        """从音频块队列创建音频文件"""
        try:
            import wave
        except ImportError:
            print("需要 wave 库来处理音频")
            return

        CHANNELS = 1
        RATE = 16000
        FORMAT = pyaudio.paInt16 if 'pyaudio' in globals() else 2  # 假设为 int16

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RATE)

            while not self.audio_queue.empty():
                data = self.audio_queue.get_nowait()
                wf.writeframes(data)


    def _send_to_openai(self, video_path, audio_path, prompt):
        """将视频和音频发送到 OpenAI API"""

        start_time_turn = time.perf_counter()
        try:
            import pyaudio
        except ImportError:
            print("需要安装 pyaudio 库来捕获音频")
            return


        try:
            video_stream = ffmpeg.input(video_path)
            audio_stream = ffmpeg.input(audio_path)
            log_level = "warning"
            process = (ffmpeg.output(
                    video_stream, audio_stream,
                    "./demo.mp4",
                    vcodec='copy',  # 直接复制视频流，不重新编码
                    acodec='aac',  # 将音频编码为 AAC 格式
                    strict='experimental'  # 允许使用实验性编解码器
                ).global_args('-loglevel', log_level)  # 设置日志级别
                .overwrite_output()  # 覆盖已存在的文件
                .run_async(pipe_stdout=True, pipe_stderr=True)  # 异步运行以获取实时输出
            )

            # 实时输出 FFmpeg 处理信息
            while True:
                stderr_line = process.stderr.readline()
                if not stderr_line:
                    break
                print(stderr_line.decode('utf-8').strip())
            process.wait()

        except Exception as e:
            print(f"合并音视频时出错: {e}")
            # 如果合并失败，保存原始视频
            if os.path.exists(video_path):
                os.rename(video_path, "./demo.mp4")
                print(f"已保存原始视频至: {"./demo.mp4"} (无音频)")


        #print("afte ... merge .........")
        with open("./demo.mp4", "rb") as f:
            video_data_out = f.read()
            video_base64_out = base64.b64encode(video_data_out).decode('utf-8')
    
        end_time = time.perf_counter()
            # 计算执行时间
        execution_time = end_time - start_time_turn
        print(f"process ...total代码执行时间: {execution_time:.6f} 秒")
            #time.sleep(1)

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "video_url",
                     "video_url": {"url": f"data:;base64,{video_base64_out}"},
                     }

                ],
            },
        ]

        start_time = time.perf_counter()
        
        import pyaudio
        p = pyaudio.PyAudio()
        # # 创建音频流
        stream = p.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=24000,
                         output=True)


        # Qwen-Omni only supports stream mode
        completion = client.chat.completions.create(
            #model="qwen-omni-turbo",
            model="qwen2.5-omni-7b",
            messages=messages,
            modalities=["text", "audio"],
            audio={
                "voice": "Ethan",  # Cherry, Ethan, Serena, Chelsie is available
                "format": "wav"
            },
            stream=True,
            stream_options={"include_usage": True}
        )

        text = []
        audio_string = ""
        count = 0



        for chunk in completion:
            count = count + 1


            if chunk.choices:
                if hasattr(chunk.choices[0].delta, "audio"):
                    try:
                        audio_string += chunk.choices[0].delta.audio["data"]
                        audio_data=chunk.choices[0].delta.audio["data"]
                        wav_bytes = base64.b64decode(audio_data)
                        wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
                        chunk_len = len(wav_array)
                        stream.write(wav_array.tobytes())
                        '''  
                        print(count)
                        end_time = time.perf_counter()
                        # 计算执行时间
                        execution_time = end_time - start_time
                        print(f"第一个chunk 从送video 到输出第一个audio执行时间: {execution_time:.6f} 秒")
                        '''
                    except Exception as e:
                        text.append(chunk.choices[0].delta.audio["transcript"])

            else:
                print(chunk.usage)

            chunk_end_time = time.perf_counter()


        import soundfile as sf
        import sounddevice as sd

        print("video understand result:")
        print("".join(text))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        #print(f"第一个chunk 从送video 到所有结果获得: {execution_time:.6f} 秒")
        
        return "Success"


    def start(self):
        """启动音视频捕获线程"""
        self.stop_event.clear()

        # 启动视频捕获线程
        self.video_thread = threading.Thread(target=self.capture_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # 启动音频捕获线程

        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop(self):
        """停止所有线程"""
        self.stop_event.set()
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=2.0)
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2.0)


# 使用示例
if __name__ == "__main__":

    API_KEY = "3e9cf6e6052a4c7dba9dd02bc30bd529"  # 替换为你的 qwen API 密钥
    stream_processor = OpenAI4oStream(api_key=API_KEY)

    try:

        # 启动捕获
        stream_processor.start()
        print("正在收集音视频数据...")
        time.sleep(context_len+3)

        while True:
            start_time_turn = time.perf_counter()
            response = stream_processor.process_stream("start the test flow")
            '''
            if response:
                print("API 响应:")
                print(response)
            '''

            end_time = time.perf_counter()
            # 计算执行时间
            execution_time = end_time - start_time_turn
            print(f"total代码执行时间: {execution_time:.6f} 秒")
            #time.sleep(1)

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 停止捕获
        stream_processor.stop()
        print("程序已停止")
