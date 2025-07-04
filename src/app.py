import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import faiss
import google.generativeai as genai
from PIL import Image
import gradio as gr
import base64
from io import BytesIO


KNOWLEDGE_FILE = "knoledge.json"
CAPTIONS_FILE = "captions_output.json"
FAISS_INDEX_FILE = "design_index.faiss"
ID_MAP_FILE = "id_map.json"

# Geminiモデル
VISION_MODEL_FOR_CAPTIONING = "gemini-1.5-flash-latest"
EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash-latest"

# --- APIキーの設定 (環境変数からのみ読み込む) ---
try:
    # 環境変数からAPIキーを読み込む。見つからない場合はエラーを発生させる。
    API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
   
    print("エラー: 環境変数 'GEMINI_API_KEY' が設定されていません。アプリを起動する前に設定してください。")
    sys.exit(1)


# --- ヘルパー関数 ---
def image_to_base64_string(pil_image, format="JPEG"):
    """Pillow ImageオブジェクトをBase64エンコードされた文字列に変換する"""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def create_focused_text(caption):
    """
    キャプションJSONから、検索に適した詳細なテキストサマリーを生成する。
    """
    if not caption:
        return ""

    parts = []

    # 1. 物品名
    article_name = caption.get('article', {}).get('name', 'N/A')
    parts.append(f"物品名: {article_name}")

    # 2. 全体印象
    holistic_impression = caption.get('form', {}).get('holisticImpression', '')
    if holistic_impression:
        parts.append(f"全体的な形状・印象: {holistic_impression}")

    # 3. 基本的な構成
    basic_composition = caption.get('form', {}).get('basicComposition', '')
    if basic_composition:
        parts.append(f"基本的な構成: {basic_composition}")

    # 4. 各構成要素の詳細
    detailed_components = caption.get('form', {}).get('detailedComponents', [])
    if detailed_components:
        component_texts = ["構成要素の詳細:"]
        for comp in detailed_components:
            name = comp.get('componentName', '無名')
            shape = comp.get('shapeDescription', '形状記述なし')
            ornamentation = comp.get('ornamentation', '装飾なし')
            comp_text = f"- {name}: 形状は「{shape}」、装飾は「{ornamentation}」"
            component_texts.append(comp_text)
        parts.append("\n".join(component_texts))

    # 5. 意匠の要部の推定
    key_features = caption.get('analysis', {}).get('hypothesizedKeyFeatures', [])
    if key_features:
        feature_texts = ["意匠の要点:"]
        for f in key_features:
            feature = f.get('feature', '特徴記述なし')
            feature_texts.append(f"- {feature}")
        parts.append("\n".join(feature_texts))

    return "\n\n".join(parts)


class DesignRAGSystem:
    """
    構築済みのデータベースを読み込み、RAG（検索拡張生成）を実行するクラス。
    """

    def __init__(self):
        """システムの初期化時に、必要なモデルとファイルをすべて読み込む。"""
        print("RAGシステムの初期化を開始します...")
        
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.captioning_model = genai.GenerativeModel(VISION_MODEL_FOR_CAPTIONING, safety_settings=safety_settings)
        self.generative_model = genai.GenerativeModel(GENERATION_MODEL, safety_settings=safety_settings)

        self._load_prompts()

        # データベースとインデックスを読み込み
        self.knowledge_dict = self._load_knowledge_base(KNOWLEDGE_FILE)
        self.captions_dict = self._load_knowledge_base(CAPTIONS_FILE, key_field='id', value_field='caption')
        
        if os.path.exists(FAISS_INDEX_FILE):
            self.index = faiss.read_index(FAISS_INDEX_FILE)
            print(f"✅ インデックス '{FAISS_INDEX_FILE}' を読み込みました。")
        else:
            raise FileNotFoundError(f"エラー: FAISSインデックスファイル '{FAISS_INDEX_FILE}' が見つかりません。")

        if os.path.exists(ID_MAP_FILE):
            with open(ID_MAP_FILE, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)
            print(f"✅ IDマップ '{ID_MAP_FILE}' を読み込みました。")
        else:
            raise FileNotFoundError(f"エラー: IDマップファイル '{ID_MAP_FILE}' が見つかりません。")
        
        print("✅ RAGシステムの初期化が完了しました。")

    def _load_knowledge_base(self, filepath, key_field='id', value_field=None):
        """汎用的なJSONローダー。IDをキーとする辞書を作成する。"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {item[key_field]: (item[value_field] if value_field else item) for item in data}
        except Exception as e:
            raise Exception(f"エラー: {filepath} の読み込みに失敗しました。 - {e}")

    def _load_prompts(self):
        """プロンプトを定義する。"""
       
        self.captioning_prompt = """
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
```
"""
        self.report_prompt = """
あなたは日本の意匠審査官として、提供された「調査対象の意匠」と「類似意匠」の情報を比較し、新規性に関する詳細な分析レポートを作成してください。

**指示:**
1.  **意匠の要点認定:** 「調査対象の意匠」の画像と情報から、意匠の要点（最も特徴的なデザイン上のポイント）を簡潔に認定してください。
2.  **類似点の指摘:** 「調査対象の意匠」と各「類似意匠」を比較し、物品の用途・機能、および形態における具体的な類似点を、根拠を示しながら詳細に指摘してください。
3.  **相違点の指摘:** 同様に、具体的な相違点を、根拠を示しながら詳細に指摘してください。
4.  **新規性の評価:** 上記の類似点と相違点を総合的に評価し、「調査対象の意匠」が新規性を有するかどうかについての見解を述べてください。相違点が創作性のレベルに達しているか、あるいはありふれた創作手法の範囲内であるかなど、評価の根拠を明確に記述してください。
5.  **結論:** 最終的な結論として、「新規性を有する可能性が高い」「新規性を有する可能性が低い」「判断が困難」のいずれかを明記し、その理由を要約してください。

**出力形式:** Markdown形式で、各項目を見出し（`##` や `###`）で区切り、論理的で読みやすいレポートを作成してください。
---
"""

    def _generate_caption(self, article_name, image_pils, article_desc, design_desc):
        """画像とテキスト情報から詳細なキャプションを生成する。"""
        prompt_parts = [self.captioning_prompt]
        prompt_parts.append(f"- **物品名:** {article_name}")
        prompt_parts.append(f"- **物品の説明:** {article_desc}")
        prompt_parts.append(f"- **意匠の説明:** {design_desc}")
        prompt_parts.extend(image_pils)

        print("キャプション生成を開始します...")
        response = self.captioning_model.generate_content(prompt_parts)
        
        try:
            match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
            if match:
                json_str = match.group(1)
                response_json = json.loads(json_str)
                print("✅ キャプション生成完了。")
                return response_json
            else:
                response_json = json.loads(response.text)
                print("✅ キャプション生成完了（直接パース）。")
                return response_json
        except (json.JSONDecodeError, AttributeError) as e:
            error_msg = f"キャプションのJSONパースに失敗しました: {e}\nモデルの生レスポンス: {response.text}"
            print(error_msg)
            raise ValueError(error_msg)

    def _embed_text(self, text):
        """テキストをembeddingベクトルに変換する。"""
        print(f"Embeddingを生成中: {text[:50]}...")
        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL}",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        print("✅ Embedding生成完了。")
        return result['embedding']

    def _search_faiss(self, embedding, k):
        """FAISSインデックスで最近傍探索を行う。"""
        print(f"FAISSインデックスを検索中 (k={k})...")
        embedding_np = np.array([embedding], dtype='float32')
        distances, indices = self.index.search(embedding_np, k)
        results = [(self.id_map[i], distances[0][j]) for j, i in enumerate(indices[0])]
        print(f"✅ FAISS検索完了。 {len(results)}件の類似意匠が見つかりました。")
        return results

    def _generate_report(self, query_data, query_image_pils, search_results, generated_caption):
        """検索結果を基に、新規性に関する分析レポートを生成する。"""
        
        model_input_contents = [self.report_prompt]
        query_info_str = "## 調査対象の意匠\n"
        query_info_str += f"- **物品名:** {query_data['article']['name']}\n"
        query_info_str += f"- **物品の説明:** {query_data['article']['article_description']}\n"
        query_info_str += f"- **意匠の説明:** {query_data['article']['design_description']}\n"
        query_info_str += f"- **生成されたキャプション:**\n```json\n{json.dumps(generated_caption, indent=2, ensure_ascii=False)}\n```"
        model_input_contents.append(query_info_str)
        model_input_contents.extend(query_image_pils)
        similar_designs_str = "\n---\n## 類似意匠の情報\n"
        similar_images_for_model = []
        for i, (doc_id, distance) in enumerate(search_results):
            knowledge_item = self.knowledge_dict.get(doc_id, {})
            caption = self.captions_dict.get(doc_id, "キャプション情報なし")
            image_path = knowledge_item.get('images', [None])[0]
            similar_designs_str += f"\n### 類似意匠 {i+1} (ID: {doc_id}, 類似度スコア: {1-distance:.2f})\n"
            similar_designs_str += f"- **物品名:** {knowledge_item.get('article', '情報なし')}\n"
            similar_designs_str += f"- **公報テキスト:** {knowledge_item.get('text', '情報なし')[:300]}...\n"
            similar_designs_str += f"- **キャプション:** {json.dumps(caption, indent=2, ensure_ascii=False)}\n"
            if image_path and os.path.exists(image_path):
                try:
                    similar_image_pil = Image.open(image_path)
                    similar_images_for_model.append(f"類似意匠 {i+1} (ID: {doc_id}) の画像:")
                    similar_images_for_model.append(similar_image_pil)
                except Exception as e:
                    print(f"類似意匠の画像読み込みエラー (ID: {doc_id}): {e}")
        model_input_contents.append(similar_designs_str)
        model_input_contents.extend(similar_images_for_model)
        print("新規性分析レポートの生成を開始します...")
        response = self.generative_model.generate_content(model_input_contents)
        report_text = response.text
        print("✅ レポート生成完了。")
        final_report_md = "## 調査対象の意匠\n"
        final_report_md += f"**物品名:** {query_data['article']['name']}\n\n"
        for pil_img in query_image_pils:
            b64_img = image_to_base64_string(pil_img)
            final_report_md += f'![調査対象画像](data:image/jpeg;base64,{b64_img})\n'
        final_report_md += f"\n---\n{report_text}\n---\n"
        final_report_md += "\n## 参考：類似意匠の詳細\n"
        for doc_id, distance in search_results:
            knowledge_item = self.knowledge_dict.get(doc_id, {})
            caption_text = json.dumps(self.captions_dict.get(doc_id, "キャプション情報なし"), indent=2, ensure_ascii=False)
            image_path = knowledge_item.get('images', [None])[0]
            final_report_md += f"\n### 類似意匠 (ID: {doc_id}) - 類似度: {1-distance:.2f}\n"
            final_report_md += f"**物品名:** {knowledge_item.get('article', '情報なし')}\n\n"
            if image_path and os.path.exists(image_path):
                try:
                    pil_img = Image.open(image_path)
                    b64_img = image_to_base64_string(pil_img)
                    final_report_md += f'![類似画像_{doc_id}](data:image/jpeg;base64,{b64_img})\n'
                except Exception as e:
                    print(f"レポートへの画像埋め込みエラー (ID: {doc_id}): {e}")
            final_report_md += f"\n**キャプション:**\n```json\n{caption_text}\n```\n"
        return final_report_md

    def perform_rag(self, query_data, k=10):
        """RAGの全プロセスを実行する。"""
        image_paths = query_data.get('images', [])
        pil_images = []
        if image_paths:
            for path in image_paths:
                try:
                    pil_images.append(Image.open(path))
                except Exception as e:
                    print(f"画像の読み込みに失敗しました: {path} - {e}")
        
        if not pil_images:
            raise ValueError("分析対象の画像がありません。")

        # 1. キャプション生成
        generated_caption = self._generate_caption(
            query_data['article']['name'],
            pil_images,
            query_data['article']['article_description'],
            query_data['article']['design_description']
        )
        if "error" in generated_caption:
            raise ValueError(generated_caption["error"])

        # 2. Embedding生成
        focused_text = create_focused_text(generated_caption)
        embedding = self._embed_text(focused_text)

        # 3. FAISS検索
        search_results = self._search_faiss(embedding, k)

        # 4. レポート生成
        report = self._generate_report(query_data, pil_images, search_results, generated_caption)
        return report

# --- Gradio UIとロジックの接続 ---
rag_system = None

def initialize_system():
    """アプリ起動時に一度だけRAGシステムを初期化する"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = DesignRAGSystem()
        except Exception as e:
            print(f"RAGシステムの初期化中にエラーが発生しました: {e}")
            # エラーが発生した場合、UIで通知できるようにする
            rag_system = None 

def rag_interface_func(image_files, article_name, article_description, design_description):
    """GradioのUIとRAGシステムを接続する関数"""
    if rag_system is None:
        # 初期化に失敗していた場合、エラーメッセージを返す
        initialize_system()
        if rag_system is None:
             return "## エラー\n\nRAGシステムの初期化に失敗しました。コンソールのエラーメッセージを確認してください。"

    try:
        if not image_files:
            error_message = "エラー: 意匠画像がアップロードされていません。"
            gr.Error(error_message)
            return ""
        
        query_data = {"article": {"name": article_name if article_name else "指定なし",
                                  "article_description": article_description if article_description else "なし",
                                  "design_description": design_description if design_description else "なし"},
                      "images": [file.name for file in image_files]}
        
        return rag_system.perform_rag(query_data, k=10)
    except Exception as e:
        error_message = f"分析中にエラーが発生しました: {e}"
        gr.Error(error_message)
        print(error_message)
        return f"## エラー\n\n{error_message}"

# --- Gradio UIの構築 ---
def build_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="意匠RAG検索・分析システム") as demo:
        gr.Markdown("# 意匠RAG検索・分析システム")
        gr.Markdown("意匠の画像と説明を入力すると、類似意匠を検索し、新規性に関する分析レポートを生成します。")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. クエリ情報入力")
                # --- APIキー入力欄を削除 ---
                image_input = gr.File(
                    label="意匠画像 (必須・複数可)",
                    file_count="multiple",
                    file_types=["image"]
                )
                article_name_input = gr.Textbox(label="物品名")
                article_description_input = gr.Textbox(label="物品の説明", lines=4)
                design_description_input = gr.Textbox(label="意匠の説明", lines=4)
                submit_btn = gr.Button("分析を実行", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 2. 分析結果")
                output_report = gr.Markdown(label="レポート")

        # --- clickイベントのinputsからapi_key_inputを削除 ---
        submit_btn.click(
            fn=rag_interface_func,
            inputs=[image_input, article_name_input, article_description_input, design_description_input],
            outputs=[output_report]
        )
        
        # Gradioアプリ起動時に一度だけシステムを初期化
        demo.load(initialize_system, None, None)
        
    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(debug=True)