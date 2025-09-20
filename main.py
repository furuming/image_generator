import argparse
import os
from typing import Tuple

import torch
from diffusers import DiffusionPipeline

# from googletrans import Translator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stable Diffusionを使用して画像を生成します')
    parser.add_argument('--height', type=int, default=256, help='生成される画像の高さ（8で割り切れる必要があります')
    parser.add_argument('--width', type=int, default=256, help='生成される画像の幅（8で割り切れる必要があります')
    parser.add_argument('--output_file', type=str, default='/data/demo.png', help='生成された画像の出力ファイル名')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto', help='実行デバイスを指定します（auto/cuda/cpu）')
    return parser.parse_args()


def validate_dimensions(height: int, width: int) -> None:
    # height と width が8で割り切れるかチェック
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"height と width は8で割り切れる値にしてください。現在の値は高さ: {height}, 幅: {width}です。")


# def translate_prompt(prompt: str) -> str:
#     # Google翻訳サービスを利用して日本語から英語に翻訳
#     translator = Translator()
#     try:
#         translated = translator.translate(prompt, src='ja', dest='en')
#         return translated.text
#     except Exception as e:
#         raise RuntimeError(f"翻訳中にエラーが発生しました: {e}")


def _select_device(requested: str) -> Tuple[str, torch.dtype]:
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDAが利用できません。NVIDIAドライバ/コンテナ設定/torch(CUDA版)を確認してください。')
        return 'cuda', torch.float16
    if requested == 'cpu':
        return 'cpu', torch.float32
    # auto
    if torch.cuda.is_available():
        return 'cuda', torch.float16
    return 'cpu', torch.float32


def _log_device_info(device: str) -> None:
    print('=== Device Info ===')
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if device == 'cuda':
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            print(f"GPU[{idx}]: {name} | VRAM: {total:.1f} GB")
        except Exception as e:
            print(f"GPU情報の取得に失敗: {e}")
    print('===================')


def generate_image(prompt: str, height: int, width: int, output_file: str, device: str) -> None:
    # デバイス選択とログ
    device, dtype = _select_device(device)
    _log_device_info(device)

    # DiffusionPipeline クラスを使用して事前学習済みモデルを読み込む
    pipe = DiffusionPipeline.from_pretrained(
        "digiplay/LusterMix_v1.5_safetensors",
        torch_dtype=dtype if device == 'cuda' else torch.float32,
    )

    # GPUが使える場合はGPUへ移動
    pipe = pipe.to(device)

    # 指定されたプロンプトと画像の高さ・幅を使用して画像を生成
    image = pipe(prompt, height=height, width=width).images[0]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    image.save(output_file)


if __name__ == "__main__":
    args = parse_arguments()
    validate_dimensions(args.height, args.width)

    user_input_prompt = input("Enter text for image generation: ")
    generate_image(user_input_prompt, args.height, args.width, args.output_file, args.device)

