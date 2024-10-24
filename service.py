import os
import sys
import argparse
import numpy as np
import torch
import scipy.io.wavfile
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import traceback

# 设置路径
ROOT_DIR = os.getcwd()
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
MATCHA_TTS_DIR = f'{ROOT_DIR}/third_party/Matcha-TTS'
if MATCHA_TTS_DIR not in sys.path:
    sys.path.append(MATCHA_TTS_DIR)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI()

max_val = 0.8
prompt_sr, target_sr = 16000, 22050  # 输出为16kHz
default_data = np.zeros(target_sr)

@app.post("/text2speech")
async def text2speech(request: Request):
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

    # 生成音频
    try:
        stream = False  # 固定为流式
        audio_chunks = []
        for i in cosyvoice.inference_instruct(text, spk_id, instruction, stream=stream, speed=speed):
            speech_chunk = i['tts_speech']
            print(speech_chunk.shape)
            audio_chunks.append(speech_chunk.numpy().flatten())

        # 拼接所有音频块
        if audio_chunks:
            full_speech = np.concatenate(audio_chunks)
            print(full_speech.shape)
        else:
            return JSONResponse(content={'error': 'No audio generated.'}, status_code=500)

        # 将音频保存到缓冲区
        buffer = BytesIO()
        # 将 float32 转换为 int16
        full_speech = np.int16(full_speech  * 32767)
        scipy.io.wavfile.write(buffer, target_sr, full_speech)
        buffer.seek(0)

        # 返回音频文件
        return StreamingResponse(buffer, media_type="audio/wav", headers={
            'Content-Disposition': 'attachment; filename="output.wav"'
        })

        # 返回音频文件
        return StreamingResponse(buffer, media_type="audio/wav", headers={
            'Content-Disposition': 'attachment; filename="output.wav"'
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