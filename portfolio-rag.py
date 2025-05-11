import os
import numpy as np # FAISSが内部で使う場合があるが、直接的な操作は減る
import google.generativeai as genai
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter # テキスト分割用
from langchain.docstore.document import Document # Documentオブジェクトを利用
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# --- 設定項目 ---
# Google Gemini APIキーの設定
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("環境変数 GOOGLE_API_KEY が設定されていません。")
genai.configure(api_key=API_KEY)

# 埋め込みモデルの指定 (HuggingFaceリポジトリ名)
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# all-MiniLM-L6-v2 の出力次元数 (参考情報、HuggingFaceEmbeddingsが自動で処理)
# EMBEDDING_DIMENSION = 384 (FAISS.from_documents を使う場合、直接指定は不要)

# テキストファイルが格納されているディレクトリ
TEXT_FILES_DIR = "text_data"  # ここに参照させたいテキストファイル (.txt, .mdなど) を置く
# FAISSインデックスを保存/ロードするパス (任意)
FAISS_INDEX_PATH = "faiss_index_text_data"

# テキスト分割の設定
CHUNK_SIZE = 1000  # 各チャンクの最大文字数
CHUNK_OVERLAP = 100 # チャンク間のオーバーラップ文字数

# --- グローバル変数 ---
# 埋め込み関数インスタンス
embeddings_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'}, # 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True} # True推奨
)
knowledge_base = None # グローバルなナレッジベースオブジェクト
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY, temperature=0.1)

# --- 関数定義 ---

def load_documents_from_text_files(text_files_dir: str) -> list[Document]:
    """
    指定されたディレクトリからテキストファイルを読み込み、Documentオブジェクトのリストとして返します。
    対応するファイル拡張子: .txt, .md
    """
    raw_documents = []
    print(f"テキストファイルディレクトリ '{text_files_dir}' からドキュメントを読み込み中...")
    if not os.path.exists(text_files_dir) or not os.listdir(text_files_dir):
        print(f"警告: ディレクトリ '{text_files_dir}' が空または存在しません。")
        return raw_documents

    supported_extensions = ('.txt', '.md')
    for file_name in os.listdir(text_files_dir):
        if file_name.lower().endswith(supported_extensions):
            file_path = os.path.join(text_files_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                raw_documents.append(Document(page_content=content, metadata={"source": file_name}))
                print(f"読み込み完了: {file_name} ({len(content)} 文字)")
            except Exception as e:
                print(f"エラー: {file_path} の読み込みに失敗しました: {e}")
    return raw_documents

def create_knowledge_base_from_text_files(text_files_dir: str, chunk_size: int, chunk_overlap: int):
    """
    テキストファイルを読み込み、チャンク化し、FAISSナレッジベースを作成します。
    """
    documents = load_documents_from_text_files(text_files_dir)
    if not documents:
        print("警告: 読み込むドキュメントがありません。ナレッジベースは作成されません。")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # チャンクの開始位置をメタデータに追加 (任意)
    )
    
    chunked_documents = text_splitter.split_documents(documents)
    print(f"{len(documents)}個のドキュメントから {len(chunked_documents)}個のチャンクを作成しました。")

    if not chunked_documents:
        print("警告: チャンク化されたドキュメントがありません。ナレッジベースは作成されません。")
        return None

    print("FAISSインデックスを作成中...")
    try:
        # FAISS.from_documents を使用してインデックスを作成
        # これにより、チャンクのテキストがベクトル化され、FAISSに格納される
        faiss_index = FAISS.from_documents(
            documents=chunked_documents,
            embedding=embeddings_function,
            # normalize_L2=True, # HuggingFaceEmbeddings側で normalize_embeddings=True を推奨
            distance_strategy='COSINE' # または 'EUCLIDEAN_L2' や 'MAX_INNER_PRODUCT'
                                       # normalize_embeddings=TrueならINNER_PRODUCTが良い
        )
        print(f"FAISSナレッジベースが {len(chunked_documents)} 件のチャンクで作成されました。")
        # オプション: 作成したインデックスを保存する
        if FAISS_INDEX_PATH:
            faiss_index.save_local(FAISS_INDEX_PATH)
            print(f"FAISSインデックスを '{FAISS_INDEX_PATH}' に保存しました。")
        return faiss_index
    except Exception as e:
        print(f"エラー: FAISSインデックスの作成に失敗しました: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def load_faiss_index(index_path: str):
    """
    保存されたFAISSインデックスをロードします。
    """
    if os.path.exists(index_path):
        try:
            # allow_dangerous_deserialization=True は LangChain の FAISS ロードで必要
            faiss_index = FAISS.load_local(index_path, embeddings_function, allow_dangerous_deserialization=True)
            print(f"FAISSインデックスを '{index_path}' からロードしました。")
            return faiss_index
        except Exception as e:
            print(f"エラー: FAISSインデックス '{index_path}' のロードに失敗しました: {e}")
            return None
    return None


def search_similar_documents(query: str, top_k: int = 3):
    """
    FAISSナレッジベースを使用して類似ドキュメントチャンクを検索します。
    """
    if knowledge_base is None:
        print("警告: ナレッジベースが初期化されていません。")
        return []

    try:
        # similarity_search_with_score は (Document, score) のリストを返す
        # Document には page_content (チャンクのテキスト) と metadata が含まれる
        # score は距離 (COSINE戦略ならコサイン距離: 0に近いほど類似)
        results_with_scores = knowledge_base.similarity_search_with_score(query, k=top_k)
        
        processed_results = []
        for doc, score in results_with_scores:
            # doc.page_content (チャンクの実際のテキスト) とスコア、ソースファイル名を返す
            processed_results.append({
                "content": doc.page_content,
                "score": score,
                "source": doc.metadata.get('source', 'N/A'),
                "start_index": doc.metadata.get('start_index', -1) # チャンク開始位置(あれば)
            })
        
        return processed_results
    except Exception as e:
        print(f"エラー: ベクトル検索中にエラーが発生しました: {e}")
        return []


def generate_answer(query: str, context_chunks: list):
    """
    検索されたコンテキストチャンクを元に、LLMで回答を生成します。
    context_chunks は search_similar_documents から返される辞書のリスト。
    """
    print(f"DEBUG generate_answer: Received query = '{query}'")
    print(f"DEBUG generate_answer: Received context_chunks = {context_chunks}")

    if not context_chunks:
        context_str = "利用可能なコンテキスト情報がありません。"
    else:
        context_str = "以下は関連性の高い情報です:\n\n"
        for i, chunk_info in enumerate(context_chunks):
            context_str += f"--- 関連情報 {i+1} (出典: {chunk_info['source']}, 関連度スコア(距離): {chunk_info['score']:.4f}) ---\n"
            context_str += chunk_info['content']
            context_str += "\n---\n\n"
    # print(f"DEBUG generate_answer: Constructed context_str = '''{context_str}'''") # 長大になる可能性

    prompt = f"""以下の情報を参考に、質問に答えてください。情報は複数の断片から構成されている場合があります。

コンテキスト情報:
{context_str}

質問: {query}

回答:"""
    
    print(f"DEBUG generate_answer: Final prompt for LLM (length: {len(prompt)}), context length: {len(context_str)}")
    if len(prompt) > 30000: # Gemini 1.5 Flash のトークン制限を考慮 (実際はもっと大きいが安全のため)
         print(f"警告: プロンプトが非常に長いです ({len(prompt)}文字)。短縮を試みます。")
         # 簡単な短縮処理（より洗練された方法が必要な場合もある）
         max_context_len = 25000 # コンテキスト情報に割り当てる最大文字数
         if len(context_str) > max_context_len:
             context_str = context_str[:max_context_len] + "\n... (コンテキスト情報が長すぎるため一部省略)"
             prompt = f"""以下の情報を参考に、質問に答えてください。情報は複数の断片から構成されている場合があります。

コンテキスト情報:
{context_str}

質問: {query}

回答:"""
             print(f"DEBUG generate_answer: 短縮後のプロンプト (length: {len(prompt)})")


    if not prompt or not prompt.strip():
        print("ERROR generate_answer: Prompt is empty or consists only of whitespace.")
        return "エラー: プロンプトが空のため、回答を生成できませんでした。"

    try:
        messages = [HumanMessage(content=prompt)]
        print(f"DEBUG generate_answer: Messages object being sent to LLM (type: {type(messages[0])}, content length: {len(messages[0].content)})")

        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"エラー: LLMでの回答生成中にエラーが発生しました: {e}")
        import traceback
        print("詳細なスタックトレース:")
        print(traceback.format_exc())
        return "申し訳ありませんが、回答を生成中にエラーが発生しました。"


def rag_system(query_text: str):
    if not query_text.strip():
        return "質問を入力してください。", "コンテキストはありません。"
    
    print(f"DEBUG rag_system: Received query_text = '{query_text}'")
        
    similar_chunks = search_similar_documents(query_text) # チャンク情報のリスト
    print(f"DEBUG rag_system: Found similar_chunks = {similar_chunks}")
    
    context_display = ""
    if similar_chunks:
        for chunk_info in similar_chunks:
            context_display += f"出典: {chunk_info['source']} (スコア: {chunk_info['score']:.4f}, 開始位置: {chunk_info.get('start_index', 'N/A')})\n"
            context_display += f"内容抜粋: {chunk_info['content'][:200]}...\n\n" # 表示用に抜粋
    else:
        context_display = "関連するドキュメントは見つかりませんでした。"
    # print(f"DEBUG rag_system: Constructed context_display = '''{context_display}'''") # 長大になる可能性
            
    answer = generate_answer(query_text, similar_chunks) # チャンク情報のリストを渡す
    print(f"DEBUG rag_system: Generated answer (first 100 chars) = '{str(answer)[:100]}...'")
    
    return answer, context_display

# --- 初期化とアプリケーション起動 ---
if __name__ == "__main__":
    # テキストファイルディレクトリのチェックと作成
    if not os.path.exists(TEXT_FILES_DIR):
        print(f"情報: ディレクトリ '{TEXT_FILES_DIR}' が存在しません。作成します。")
        try:
            os.makedirs(TEXT_FILES_DIR)
            print(f"'{TEXT_FILES_DIR}' に、参照させたいテキストファイル (.txt, .mdなど) を配置してください。")
            # 例: ダミーファイル作成
            # with open(os.path.join(TEXT_FILES_DIR, "sample_document.txt"), "w", encoding="utf-8") as f:
            #     f.write("これはサンプルドキュメントです。AIと機械学習について書かれています。\n")
            #     f.write("LangChainはLLMアプリケーション開発を容易にするフレームワークです。")
        except OSError as e:
            print(f"エラー: ディレクトリ '{TEXT_FILES_DIR}' の作成に失敗しました: {e}")
            exit(1)

    # ナレッジベースの初期化
    # まず保存されたインデックスのロードを試みる
    if FAISS_INDEX_PATH:
        knowledge_base = load_faiss_index(FAISS_INDEX_PATH)

    # ロードできなかった場合、またはパスが指定されていない場合は新規作成
    if knowledge_base is None:
        print("保存されたFAISSインデックスが見つからないかロードに失敗したため、新規に作成します。")
        knowledge_base = create_knowledge_base_from_text_files(
            TEXT_FILES_DIR,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

    if knowledge_base is None:
        print("致命的エラー: ナレッジベースの作成またはロードに失敗しました。プログラムを終了します。")
        # Gradio UI を起動せずに終了するか、限定的な動作をさせるか
        # ここでは終了する例
        # exit(1)
        # または、Gradioでエラーメッセージを表示するようにする
        # interface = gr.Interface(fn=lambda x: ("ナレッジベースの初期化に失敗しました。", "エラー"), ...) で起動
        print("Gradio UI は起動しますが、ナレッジベースが機能しないため、RAGシステムは正しく動作しません。")


    # Gradio UI
    interface = gr.Interface(
        fn=rag_system,
        inputs=gr.Textbox(lines=3, placeholder="質問を入力してください...", label="質問"),
        outputs=[
            # gr.Textbox(label="回答", lines=15, show_copy_button=True),
            gr.Markdown(label="回答"),
            gr.Textbox(label="参照されたコンテキストの抜粋 (出典、スコア、内容冒頭)", lines=10)
        ],
        title="〇〇さん 面接前評価システム テキストファイル参照型RAGシステム (Google Gemini & FAISS)",
        description=(
            f"'{TEXT_FILES_DIR}' ディレクトリ内のテキストファイルの内容を参照して回答を生成します。\n"
            "コンテキストのスコアは距離を示します (小さいほど関連性が高いです)。\n"
            f"使用埋め込みモデル: {EMBEDDING_MODEL_NAME}\n"
            f"チャンク設定: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
        ),
        examples=[
            ["〇〇とはどのような人物ですか？"],
            ["募集職種についての適性を教えてください"]
        ],
        allow_flagging='never'
    )

    print("Gradioアプリケーションを起動します...")
    interface.launch()