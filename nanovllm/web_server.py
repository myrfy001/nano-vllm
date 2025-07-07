from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import time
import json
import random
from threading import Thread

import queue

model_req_queue = queue.Queue()
model_resp_queue = queue.Queue()

app = Flask(__name__)
CORS(app)  # 启用CORS支持

def generate_ai_response(prompt):
    """更真实的模拟响应生成器"""
    responses = [
        f"I understand you're asking about {prompt}. ",
        "Let me think about that... ",
        "This is an interesting question. ",
        "Based on my knowledge, ",
        "I would suggest considering multiple perspectives. ",
        "The answer might depend on several factors. ",
        "Have you thought about alternative approaches? ",
        "In many cases, the solution emerges from careful analysis. ",
        "Would you like me to elaborate further on this topic? ",
        "I hope this information is helpful for your needs."
    ]
    
    # 打乱响应顺序并组合
    random.shuffle(responses)
    return [word for sentence in responses[:random.randint(3, 6)] for word in sentence.split()]


@app.route('/', methods=['GET'])
def show_home_page():
    return Response(HOME_PAGE_HTML)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    model_req_queue.put(prompt)
    
    def generate():
        while True:
            word = model_resp_queue.get()
            if word is None:
                break
            
            event_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(event_data)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def run_web_server_in_thread():
    def inner():
        app.run(host='0.0.0.0', port=5000, threaded=True)
    
    thread = Thread(target=inner)
    thread.start()



HOME_PAGE_HTML=r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek Chat</title>
    <style>
        :root {
            --primary-color: #6e48aa;
            --secondary-color: #9d50bb;
            --background-color: #f5f7fa;
            --text-color: #333;
            --light-text: #777;
            --chat-bg-user: #f0f4ff;
            --chat-bg-ai: #ffffff;
            --border-color: #e1e4e8;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        header {
            display: flex;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        
        .logo {
            width: 40px;
            height: 40px;
            margin-right: 15px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .message {
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            max-width: 80%;
        }
        
        .user-message {
            background-color: var(--chat-bg-user);
            margin-left: auto;
            border-top-right-radius: 0;
        }
        
        .ai-message {
            background-color: var(--chat-bg-ai);
            margin-right: auto;
            border-top-left-radius: 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .input-container {
            display: flex;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
        }
        
        #user-input {
            flex: 1;
            padding: 12px 15px;
            border: 1px solid var(--border-color);
            border-radius: 20px;
            font-size: 16px;
            outline: none;
            transition: border 0.3s;
        }
        
        #user-input:focus {
            border-color: var(--primary-color);
        }
        
        #send-button {
            margin-left: 10px;
            padding: 12px 20px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 16px;
            transition: opacity 0.3s;
        }
        
        #send-button:hover {
            opacity: 0.9;
        }
        
        #send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            display: flex;
            padding: 10px 15px;
            align-items: center;
            color: var(--light-text);
            font-style: italic;
        }
        
        .typing-dots {
            display: flex;
            margin-left: 8px;
        }
        
        .typing-dot {
            width: 6px;
            height: 6px;
            background-color: var(--light-text);
            border-radius: 50%;
            margin: 0 2px;
            animation: typingAnimation 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typingAnimation {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-5px);
            }
        }
        
        .token-counter {
            font-size: 12px;
            color: var(--light-text);
            text-align: right;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">DS</div>
            <h1>DeepSeek Chat</h1>
        </header>
        
        <div class="chat-container" id="chat-container">
            <!-- Messages will be added here dynamically -->
        </div>
        
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Type your message here..." autocomplete="off">
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chat-container');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            
            userInput.focus();
            
            sendButton.addEventListener('click', sendMessage);
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;
                
                addMessage('user', message);
                userInput.value = '';
                userInput.disabled = true;
                sendButton.disabled = true;
                
                const typingElement = showTypingIndicator();
                
                getAIResponse(message, typingElement);
            }
            
            function addMessage(sender, content) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = content;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function showTypingIndicator() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message ai-message typing-indicator';
                typingDiv.innerHTML = `
                    <span>DeepSeek is typing</span>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                `;
                chatContainer.appendChild(typingDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return typingDiv;
            }
            
            function removeTypingIndicator(typingElement) {
                if (typingElement && typingElement.parentNode) {
                    typingElement.parentNode.removeChild(typingElement);
                }
            }
            
            async function getAIResponse(prompt, typingElement) {
                let aiMessageDiv = null;
                let fullResponse = '';
                let tokenCount = 0;
                
                try {
                    const response = await fetch('http://localhost:5000/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ prompt: prompt })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    // 创建AI消息容器
                    aiMessageDiv = document.createElement('div');
                    aiMessageDiv.className = 'message ai-message';
                    chatContainer.appendChild(aiMessageDiv);
                    
                    // 创建token计数器
                    const tokenCounter = document.createElement('div');
                    tokenCounter.className = 'token-counter';
                    aiMessageDiv.appendChild(document.createTextNode('')); // 空文本节点用于内容
                    aiMessageDiv.appendChild(tokenCounter);
                    
                    // 使用TextDecoder处理流数据
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.substring(6).trim();
                                if (data === '[DONE]') {
                                    break;
                                }
                                
                                try {
                                    const parsed = JSON.parse(data);
                                    if (parsed.choices && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                                        const token = parsed.choices[0].delta.content;
                                        fullResponse += token;
                                        tokenCount++;
                                        
                                        // 更新消息内容
                                        aiMessageDiv.childNodes[0].textContent = fullResponse;
                                        tokenCounter.textContent = `${tokenCount} tokens`;
                                        
                                        // 滚动到底部
                                        chatContainer.scrollTop = chatContainer.scrollHeight;
                                    }
                                } catch (e) {
                                    console.error('Error parsing JSON:', e, 'Data:', data);
                                }
                            }
                        }
                    }
                    
                } catch (error) {
                    console.error('Error:', error);
                    if (!aiMessageDiv) {
                        aiMessageDiv = document.createElement('div');
                        aiMessageDiv.className = 'message ai-message';
                        chatContainer.appendChild(aiMessageDiv);
                    }
                    aiMessageDiv.textContent = 'Sorry, an error occurred while generating the response.';
                } finally {
                    removeTypingIndicator(typingElement);
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                }
            }
        });
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    run_web_server_in_thread()