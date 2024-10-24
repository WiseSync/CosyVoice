import os
import sys
import argparse
import numpy as np
import opuslib  # 导入 opuslib 用于 Opus 编码
import torchaudio  # 导入 torchaudio 用于重采样
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import traceback
import torch

# 设置路径
ROOT_DIR = os.getcwd()
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
MATCHA_TTS_DIR = f'{ROOT_DIR}/third_party/Matcha-TTS'
if MATCHA_TTS_DIR not in sys.path:
    sys.path.append(MATCHA_TTS_DIR)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI()

# 全局配置
prompt_sr, target_sr = 16000, 16000  # 输出采样率为16kHz
default_data = np.zeros(target_sr)

resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=target_sr)

# 初始化 Opus 编码器
# 参数：
# - 采样率（16000 Hz）
# - 声道数（1声道）
# - 应用类型（APPLICATION_AUDIO）
try:
    opus_encoder = opuslib.Encoder(target_sr, 1, opuslib.APPLICATION_AUDIO)
    # 可选：设置 Opus 编码器参数，如比特率、复杂度等
    opus_encoder.bitrate = 24000  # 设置比特率为64kbps
    opus_encoder.complexity = 10    # 设置编码复杂度
except Exception as e:
    print("Opus 编码器初始化失败：", e)
    sys.exit(1)

async def audio_generator(text, instruction, spk_id, speed):
    """
    生成器函数，逐步生成并发送 Opus 编码的音频数据块。
    """
    try:
        stream = True  # 固定为流式
        buffer = np.array([], dtype=np.int16)  # 初始化缓冲区

        for i in cosyvoice.inference_instruct(text, spk_id, instruction, stream=stream, speed=speed):
            speech_chunk = i['tts_speech']

            with torch.no_grad():
                speech_resampled = resampler(speech_chunk)  # (1, M)

            speech_clamped = torch.clamp(speech_resampled, -1.0, 1.0)
            speech_int16 = (speech_clamped * 32767).to(torch.int16)  # (M,)
                
            speech_np = speech_int16.numpy().flatten()

            # 将下采样后的音频数据添加到缓冲区
            buffer = np.concatenate((buffer, speech_np))

            # 每次取出320帧进行编码
            while len(buffer) >= 320:
                frame = buffer[:320]
                buffer = buffer[320:]

                # 编码为 Opus 数据
                opus_data = opus_encoder.encode(frame.tobytes(), 320)
                # 发送 Opus 数据长度（16位）
                yield len(opus_data).to_bytes(2, byteorder='little')
                # 发送 Opus 数据
                yield opus_data

        # 处理缓冲区中剩余的音频数据
        if len(buffer) > 0:
            # 如果剩余帧不足320帧，进行填充
            padding = 320 - len(buffer)
            frame = np.pad(buffer, (0, padding), 'constant', constant_values=0)
            opus_data = opus_encoder.encode(frame.tobytes(), 320)
            # 发送 Opus 数据长度（16位）
            yield len(opus_data).to_bytes(2, byteorder='little')
            yield opus_data

    except Exception as e:
        traceback.print_exc()
        yield b''  # 发送空数据表示结束

@app.post("/text2speech")
async def text2speech(request: Request):
    """
    处理 /text2speech 的 POST 请求，生成音频并流式传输给客户端。
    """
    try:
        data = await request.json()
    except:
        traceback.print_exc()
        return JSONResponse(content={'error': 'Invalid JSON data.'}, status_code=400)

    text = data.get('text', '')
    instruction = data.get('instruction', '')
    seed = data.get('seed', 0)
    spk_id = data.get('spk_id', '')
    speed = data.get('speed', None)

    # 验证必要参数
    if not text:
        return JSONResponse(content={'error': 'Text is required.'}, status_code=400)
    if not instruction:
        return JSONResponse(content={'error': 'Instruction is required.'}, status_code=400)
    if not spk_id:
        return JSONResponse(content={'error': 'spk_id is required.'}, status_code=400)
    if speed is None:
        return JSONResponse(content={'error': 'speed is required.'}, status_code=400)

    # 验证 spk_id 是否有效
    available_spks = cosyvoice.list_avaliable_spks()
    print("Available speaker IDs:", available_spks)  # 打印可用的发音人列表
    if spk_id not in available_spks:
        return JSONResponse(content={'error': f"Invalid spk_id. Available spk_ids: {available_spks}"}, status_code=400)

    # 验证 speed 参数
    try:
        speed = float(speed)
        if speed <= 0:
            return JSONResponse(content={'error': 'Invalid speed value. It must be a positive number.'}, status_code=400)
    except ValueError:
        return JSONResponse(content={'error': 'Invalid speed value. It must be a number.'}, status_code=400)

    # 设置随机种子
    set_all_random_seed(seed)

    # 返回 StreamingResponse
    try:
        generator = audio_generator(text, instruction, spk_id, speed)
        return StreamingResponse(generator, media_type="audio/opus", headers={
            'Content-Disposition': 'attachment; filename="output.opus"'
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={'error': str(e)}, status_code=500)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M-Instruct',
                        help='模型的本地路径或 ModelScope 仓库 ID')
    args = parser.parse_args()

    # 初始化 CosyVoice 模型
    cosyvoice_model_dir = args.model_dir
    global cosyvoice
    cosyvoice = CosyVoice(cosyvoice_model_dir)
    global sft_spk
    sft_spk = cosyvoice.list_avaliable_spks()
    print("Available speaker IDs:", sft_spk)  # 打印可用的发音人列表
    global default_data
    default_data = np.zeros(target_sr)

    # 启动服务器
    uvicorn.run(app, host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()