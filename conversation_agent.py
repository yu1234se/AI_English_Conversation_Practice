from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM設定
LM_STUDIO_LLM_API_BASE = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "not-needed"
LM_STUDIO_LLM_MODEL_NAME = "EZO2.5-gemma-3-12b-it-Preview.Q6_K.gguf"

# 会話プロンプトテンプレート
CONVERSATION_TEMPLATE = """
あなたは、プロの英会話講師です。
以下の制約条件と入力文をもとに、最高の英会話レッスンをしてください。

#制約条件:
・英語での会話をロールプレイングしてください。
・1つずつ英語でのやりとりを行ってください。
・まずはあなたから会話を始めてください。
・出力は基本的には英語の会話だけでお願いします。
・余計なことを言わずになりきって会話をしてください。
・文法や表現が適切でない場合のみ日本語で指摘してください。

#入力文:
以下の条件に合わせた英会話ロープレをお願いします。
・ジャンル:日常英会話
・シチュエーション:海外旅行
・レベル:TOEIC600点レベル

Current conversation:
{history}

User: {input}
Assistant:
"""

def get_llm():
    """LLMインスタンスを取得（キャッシュ付き）"""
    return ChatOpenAI(
        openai_api_base=LM_STUDIO_LLM_API_BASE,
        openai_api_key=LM_STUDIO_API_KEY,
        model_name=LM_STUDIO_LLM_MODEL_NAME,
        temperature=0.7,
        max_tokens=100
    )

def generate_response(user_input, conversation_history):
    """英会話応答を生成"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(CONVERSATION_TEMPLATE)
    
    # 会話履歴を文字列に変換
    history_str = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation_history[-6:]
    )
    
    chain = prompt | llm
    response = chain.invoke({"input": user_input, "history": history_str})
    return response.content.strip()