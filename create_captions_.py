import os
import json
import re
import sys
import time  # timeモジュールをインポート
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image

# --- APIキーの設定 ---
# 環境変数からAPIキーを読み込む
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("エラー: 環境変数 'GEMINI_API_KEY' が設定されていません。")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# --- プロンプトの定義 (変更なし) ---
PROMPT_TEMPLATE = """
役割と目的

あなたは、日本の意匠法を専門とする特許分析のエキスパートです。あなたの任務は、入力された意匠登録出願の情報に基づき、先行意匠調査のためのRetrieval-Augmented Generation (RAG) システムで使用される、詳細かつ構造化されたJSON形式のキャプションを生成することです。生成されるキャプションは、日本の意匠法における類否判断の枠組み（物品の類否、形態の類否）に厳密に従う必要があります。

入力仕様

ユーザーから以下の情報が提供されます。
images: 意匠の様々な角度からの図面を示す画像ファイルの配列。
article_name: 「意匠に係る物品」の名称（文字列）。
article_description: 「意匠に係る物品の説明」（文字列）。
design_description:「意匠の説明」(文字列)。

実行手順

以下のステップに従い、思考を巡らせながら分析を実行し、最終的に単一のJSONオブジェクトを出力してください。
ステップ1：三次元形状の統合的理解
まず、提供されたすべてのimagesとarticle_descriptionを統合的に解釈し、対象物の完全な三次元形状、構造、および質感を把握してください。これは後のすべての分析の基礎となります。
ステップ2：物品の分析 (articleオブジェクトの生成)
次に、article_nameとarticle_descriptionを詳細に分析し、以下の情報をarticleオブジェクトにまとめてください。
name: article_nameをそのまま記載します。
useAndFunction: 物品の具体的な使用目的、使用方法、機能、および使用される状況や環境について、説明文から読み取れる情報を基に詳細なパラグラフを生成してください。
category: 物品の用途・機能に基づき、最も適切と考えられる一般的なカテゴリ（例：「調理器具」「自動車部品」「情報通信機器」「文房具」など）を特定してください。
ステップ3：形態の分析 (formオブジェクトの生成)
ステップ1で構築した三次元モデルに基づき、意匠の形態を詳細に分析し、以下の情報をformオブジェクトにまとめてください。
holisticImpression: 意匠全体を観察したときに受ける、統一的な美的印象を記述するパラグラフを生成してください（例：「ミニマルで幾何学的な印象」「有機的で流れるようなフォルム」「堅牢で工業的な雰囲気」など）。
basicComposition: 意匠を構成する主要な要素（部品）は何か、またそれらがどのように配置されているか、意匠の基本的な骨格を説明してください。
detailedComponents: 意匠を構成する各要素について、オブジェクトの配列として詳細に記述してください。各オブジェクトには以下のフィールドを含めます。
componentName: 要素の名称（例：「把手部」「注ぎ口」「本体胴部」）。
shapeDescription: その要素の具体的な幾何学的形状（例：「円筒形」「上面がわずかに湾曲した直方体」など）を詳細に記述してください。
ornamentation: 表面に施された模様、テクスチャ、仕上げ、装飾などがあれば記述してください（例：「ヘアライン仕上げ」「菱形のエンボス加工」「透明な窓」など）。なければ「なし」と記載してください。
relationships: 他の要素とどのように接続・配置されているか、その空間的・構造的関係を記述してください（例：「本体上部に溶接されている」「胴部側面から一体的に突出している」）。
ステップ4：法的・分析的推論 (analysisオブジェクトの生成)
最後に、ここまでの分析に基づき、専門家としての推論を行ってください。以下の情報をanalysisオブジェクトにまとめてください。
targetObserver: この物品の典型的な「需要者」（取引者、最終消費者など）は誰かを推論してください（例：「一般消費者」「プロの料理人」「建設業者」）。
hypothesizedKeyFeatures: この意匠の美的価値の核心をなし、需要者の注意を最も強く引くと考えられる、創造的に重要と思われる特徴（意匠の要部と推定される部分）を2～4点挙げてください。オブジェクトの配列とし、各オブジェクトには以下のフィールドを含めます。
feature: 特徴の簡潔な名称（例：「螺旋状にねじれた把手」）。
justification: なぜそれが要部と推定されるのか、その理由（新規性、独創性、視覚的支配性など）を簡潔に説明してください。
commonFeatures: この種の物品において、一般的、ありふれている、または機能上不可欠と考えられる形態的特徴をリストアップしてください。

出力形式

最終的な出力は、必ず単一の有効なJSONオブジェクトでなければなりません。JSONオブジェクトの前後に説明文などを一切含めないでください。
"""


def parse_design_text(text):
    """'text'フィールドから必要な情報を抽出する (変更なし)"""
    article_match = re.search(r'（５４）【意匠に係る物品】\n(.*?)\n', text)
    article_name = article_match.group(1).strip() if article_match else "N/A"

    article_desc_match = re.search(r'（５５）【意匠に係る物品の説明】\n(.*?)(?=\n【図面】|\n（５５）【意匠の説明】|\n（５６）|$)', text, re.DOTALL)
    article_description = article_desc_match.group(1).strip() if article_desc_match else "なし"

    design_desc_match = re.search(r'（５５）【意匠の説明】\n(.*?)(?=\n【図面】|\n（５６）|$)', text, re.DOTALL)
    design_description = design_desc_match.group(1).strip() if design_desc_match else "なし"
    
    return article_name, article_description, design_description

def clean_json_response(response_text):
    """Geminiからの応答をクリーンなJSON文字列に変換する (変更なし)"""
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        return match.group(1)
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        return response_text[first_brace:last_brace+1]
    return response_text

def create_caption_for_design(design_data):
    """単一の意匠データに対してキャプションを生成する (変更なし)"""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    article_name, article_description, design_description = parse_design_text(design_data.get("text", ""))
    image_objects = []
    image_paths_raw = design_data.get("images", [])
    if not image_paths_raw:
        # 画像がない場合は処理しない
        return None
        
    for img_path_raw in image_paths_raw:
        img_path_corrected = Path(img_path_raw.replace("design_rights/", "./"))
        try:
            if img_path_corrected.is_file():
                image_objects.append(Image.open(img_path_corrected))
        except Exception:
            pass
            
    if not image_objects:
        return None

    prompt_parts = [
        PROMPT_TEMPLATE,
        *image_objects,
        f"article_name: {article_name}",
        f"article_description: {article_description}",
        f"design_description: {design_description}",
    ]

    try:
        response = model.generate_content(prompt_parts)
        cleaned_response = clean_json_response(response.text)
        caption_json = json.loads(cleaned_response)
        return caption_json
    except Exception as e:
        print(f"\nID {design_data.get('id', 'N/A')} のキャプション生成中にエラーが発生しました: {e}")
        return None

def main(input_filepath, output_filepath):
    """メイン処理 (レート制限と再開処理を考慮)"""
    # 1. 全ての意匠データを読み込む
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            all_designs = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"エラー: {input_filepath} は有効なJSONファイルではありません。")
        return

    # 2. 既存の出力ファイルを読み込み、処理済みIDのセットを作成
    processed_captions = {}
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                # リストをIDをキーとする辞書に変換
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    for item in existing_data:
                        if 'id' in item:
                            processed_captions[item['id']] = item
        except (json.JSONDecodeError, IOError):
            print(f"警告: {output_filepath} が空か不正な形式です。新規に作成します。")
    
    processed_ids = set(processed_captions.keys())
    print(f"情報: {len(processed_ids)}件の処理済みキャプションを読み込みました。")

    # 3. 未処理の意匠データを処理
    # tqdmを使って進捗バーを表示
    for design in tqdm(all_designs, desc="キャプション生成中"):
        design_id = design.get("id")
        
        # 処理済みであればスキップ
        if design_id in processed_ids:
            continue

        print(f"\n\n▶ ID: {design_id} のキャプションを生成します。")
        caption = create_caption_for_design(design)

        if caption:
            # 成功したら結果を辞書に追加
            processed_captions[design_id] = {
                "id": design_id,
                "caption": caption
            }
            print(f"✓ ID: {design_id} のキャプションを生成しました。")
            
            # -----------------------------------
            # ★進捗をファイルに逐次保存
            # -----------------------------------
            try:
                # 辞書の値をリストに変換して保存
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(list(processed_captions.values()), f, indent=2, ensure_ascii=False)
                # print(f"  進捗を {output_filepath} に保存しました。")
            except IOError as e:
                print(f"エラー: ファイルへの書き込みに失敗しました: {e}")


        # -----------------------------------
        # ★レート制限のための待機
        # -----------------------------------
        # 無料ならGemini FlashモデルのRPM 3を考慮し、20秒+バッファ1秒で21秒待機
        wait_time = 1
        print(f"レート制限のため {wait_time} 秒待機します...", end="")
        for _ in range(wait_time):
            time.sleep(1)
            print(".", end="", flush=True)
        print(" 再開します。")


    print(f"\n\n🎉 処理が完了しました！合計{len(processed_captions)}件のキャプションを {output_filepath} に保存しました。")


if __name__ == "__main__":
    # 入力ファイルと出力ファイルを指定
    input_file = "knoledge.json"
    output_file = "captions_output.json" # 添付ファイルと同じ名前
    
    # 処理を実行
    main(input_file, output_file)