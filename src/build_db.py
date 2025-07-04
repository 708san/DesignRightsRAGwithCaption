import os
import json
import numpy as np
import faiss
import google.generativeai as genai

# --- 定数と設定 ---
CAPTIONS_FILE = "captions_output.json"
EMBEDDING_MODEL = "text-embedding-004"
FAISS_INDEX_FILE = "design_index.faiss"
ID_MAP_FILE = "id_map.json"

# --- APIキーの設定 ---
try:
    API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
    raise ValueError("エラー: 環境変数 'GEMINI_API_KEY' が設定されていません。")

#embeddingに使う文章を生成
def create_focused_text(caption):
    """
    キャプションJSONから、検索に適した詳細なテキストサマリーを生成する。
    この関数は、検索アプリ側でも全く同じものを使用する。
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

def build_and_save_database():
    """意匠の要部に焦点を当てたベクトルデータベースを構築し、保存する"""
    print("データベース構築プロセスを開始します...")

    try:
        with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        print(f"エラー: {CAPTIONS_FILE} の読み込みに失敗しました。 - {e}")
        return

    texts_to_embed = []
    id_map = []
    for item in captions_data:
        caption = item.get("caption")
       
        focused_text = create_focused_text(caption)
        if focused_text:
            texts_to_embed.append(focused_text)
            id_map.append(item['id'])
    
    if not texts_to_embed:
        print("エラー: 埋め込み対象のテキストが生成できませんでした。")
        return

    print(f"{len(texts_to_embed)}件のキャプションをベクトル化します...")
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts_to_embed,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = np.array(result['embedding'])

    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    index.add_with_ids(embeddings, np.array(range(len(id_map))).astype('int64'))
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ インデックスを '{FAISS_INDEX_FILE}' に保存しました。")
    
    with open(ID_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    print(f"✅ IDマップを '{ID_MAP_FILE}' に保存しました。")

if __name__ == "__main__":
    build_and_save_database()