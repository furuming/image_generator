import argparse
from diffusers import DiffusionPipeline
# from googletrans import Translator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stable Diffusionを使用して画像を生成します')
    parser.add_argument('--height', type=int, default=256, help='生成される画像の高さ（8で割り切れる必要があります')
    parser.add_argument('--width', type=int, default=256, help='生成される画像の幅（8で割り切れる必要があります')
    parser.add_argument('--output_file', type=str, default='demo.png', help='生成された画像の出力ファイル名')
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


def generate_image(prompt: str, height: int, width: int, output_file: str) -> None:
    # DiffusionPipeline クラスを使用して事前学習済みモデルを読み込む
    pipe = DiffusionPipeline.from_pretrained("digiplay/LusterMix_v1.5_safetensors")
    # 指定されたプロンプトと画像の高さ・幅を使用して画像を生成
    image = pipe(prompt, height=height, width=width).images[0]
    image.save(output_file)


if __name__ == "__main__":
    args = parse_arguments()
    validate_dimensions(args.height, args.width)

    user_input_prompt = input("Enter text for image generation: ")
    generate_image(user_input_prompt, args.height, args.width, args.output_file)

