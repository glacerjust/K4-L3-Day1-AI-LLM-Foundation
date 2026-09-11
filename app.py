import os
import gradio as gr
from openai import OpenAI

# Tái sử dụng các cấu hình và hàm tiện ích từ file template.py
from template import OPENAI_MODEL, retry_with_backoff

# Khởi tạo client OpenAI (api key tự động được load từ .env nhờ template.py)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_assistant_web(message, history, persona, max_turns, enable_memory, temperature, top_p):
    # Nếu có giới hạn số lượt và số lượt hiện tại đã vượt qua giới hạn
    current_turns = len(history) // 2
    if max_turns > 0 and current_turns >= max_turns:
        yield f"Đã đạt giới hạn tối đa {int(max_turns)} lượt hội thoại. Vui lòng tải lại trang (F5) để bắt đầu phiên mới."
        return

    # Gradio 5+ truyền history dưới dạng mảng các dict (1 lượt = 2 dicts)
    # Lấy 6 messages gần nhất (tương đương 3 lượt)
    recent_history = history[-6:] if len(history) > 6 else history
    
    # --- Tóm tắt ngữ cảnh ---
    if enable_memory and len(history) > 6:
        old_history = history[:-6]
        chat_log = ""
        for msg in old_history:
            role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
            content = msg.get("content", "")
            chat_log += f"{role}: {content}\n"
            
        summary_prompt = (
            "Hãy tóm tắt thật ngắn gọn (1-2 câu) nội dung chính của cuộc trò chuyện cũ sau đây "
            "để làm bối cảnh (context) cho các câu hỏi tiếp theo.\n\n" + chat_log
        )
        
        try:
            summary_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}]
            )
            summary_text = summary_response.choices[0].message.content
            persona += f"\n\n[Bối cảnh cuộc trò chuyện trước đó: {summary_text}]"
        except Exception as e:
            print("Lỗi khi tóm tắt ngữ cảnh:", e)
    # ------------------------

    messages = [{"role": "system", "content": persona}]
    
    for msg in recent_history:
        messages.append({
            "role": msg.get("role", "user"), 
            "content": msg.get("content", "")
        })
            
    if isinstance(message, dict):
        messages.append({"role": "user", "content": message.get("content", "")})
    else:
        messages.append({"role": "user", "content": message})
    def call_api():
        return client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p
        )

    # Sử dụng hàm retry chống đứt kết nối
    stream = retry_with_backoff(call_api)

    reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        reply += delta
        # Trả từng luồng kết quả về phía giao diện web
        yield reply

# Khởi tạo Giao diện Web với Gradio
with gr.Blocks(title="AI Assistant Web") as demo:
    gr.Markdown("# 🤖 Trợ lý AI - Web Interface")
    gr.Markdown("Giao diện Localhost đơn giản sử dụng Gradio, xây dựng trên nền hàm `run_assistant_web`.")
    
    with gr.Row():
        persona_input = gr.Textbox(
            label="System Prompt (Persona)", 
            value="Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.", 
            lines=2
        )
        max_turns_input = gr.Number(
            label="Giới hạn số lượt chat (0 = không giới hạn)", 
            value=0, 
            precision=0
        )
        enable_memory_input = gr.Checkbox(
            label="Bật tính năng Tóm tắt ngữ cảnh (Memory)",
            value=True
        )
        temperature_input = gr.Slider(
            label="Temperature",
            minimum=0.0, maximum=2.0, step=0.1, value=1.0
        )
        top_p_input = gr.Slider(
            label="Top P",
            minimum=0.0, maximum=1.0, step=0.05, value=1.0
        )
        
    # Tích hợp hàm run_assistant_web vào ChatInterface của Gradio
    gr.ChatInterface(
        fn=run_assistant_web,
        additional_inputs=[persona_input, max_turns_input, enable_memory_input, temperature_input, top_p_input],
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Nhập câu hỏi của bạn vào đây...", container=False, scale=7)
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
