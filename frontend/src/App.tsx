import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './FileUpload';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  isError?: boolean;
}

interface CurrentFile {
  filename: string;
  content: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://takku-ai-production.up.railway.app/api/v1';

const TakkuChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentFile, setCurrentFile] = useState<CurrentFile | null>(null);
  const [fileUploadMode, setFileUploadMode] = useState(false);
  const [suggestions] = useState([
    "What's the best way to learn programming?",
    "Tell me a fun fact about space!",
    "How can I be more productive?",
    "What are some good books to read?",
    "Explain quantum computing in simple terms",
    "What's your favorite superhero movie?"
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleFileProcessed = (filename: string, content: string) => {
    setCurrentFile({ filename, content });
    setFileUploadMode(true);
    
    const fileMessage: Message = {
      role: 'assistant',
      content: `📁 I've loaded "${filename}"! Now you can ask me questions about this document. What would you like to know? 🐱`
    };
    setMessages(prev => [...prev, fileMessage]);
  };

  const exitFileMode = () => {
    setCurrentFile(null);
    setFileUploadMode(false);
    
    const exitMessage: Message = {
      role: 'assistant', 
      content: "Switched back to regular chat mode! What would you like to talk about? 🐱"
    };
    setMessages(prev => [...prev, exitMessage]);
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: input };
    const updatedMessages = [...messages, userMessage];
    
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);

    try {
      let endpoint = `${API_BASE_URL}/ask`;
      let requestData: any = {
        question: input,
        conversation_history: updatedMessages.map(msg => ({ role: msg.role, content: msg.content }))
      };

      if (fileUploadMode && currentFile) {
        endpoint = `${API_BASE_URL}/ask`;
        requestData.question = `About the file "${currentFile.filename}": ${input}`;
      }

      const response = await axios.post(endpoint, requestData);

      const aiMessage: Message = { 
        role: 'assistant', 
        content: response.data.answer,
        model: response.data.model
      };
      
      setMessages([...updatedMessages, aiMessage]);
    } catch (error) {
      const errorMessage: Message = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.',
        isError: true
      };
      setMessages([...updatedMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="takku-chat-container">
      {/* Header */}
      <div className="chat-header">
        <div className="logo-title">
          <img src={require('./assets/takku-logo.png')} alt="Takku Logo" className="logo" />
          <div>
            <h1>Takku your AI bud</h1>
            <p>Ask anything - Your friendly AI companion</p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h3>Welcome to Takku AI!</h3>
            <p>Ask me anything! I'm your friendly AI bud here to help with questions, advice, or just chat!</p>
            
            {/* File Upload Section */}
            <div className="file-upload-section">
              <h4>📁 Or upload a document:</h4>
              <FileUpload onFileProcessed={handleFileProcessed} />
            </div>

            <div className="suggestions">
              <h4>Try asking:</h4>
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  className="suggestion-btn"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* File Mode Indicator */}
        {fileUploadMode && currentFile && (
          <div className="file-mode-indicator">
            <div className="file-info">
              <span className="file-icon">📁</span>
              <span className="file-name">{currentFile.filename}</span>
              <button className="exit-file-mode" onClick={exitFileMode}>
                ✕ Exit File Mode
              </button>
            </div>
            <div className="file-preview">
              <strong>Preview:</strong> {currentFile.content}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role} ${message.isError ? 'error' : ''}`}
          >
            <div className="message-content">
              <div className="message-text">{message.content}</div>
              {message.model && (
                <div className="message-model">Generated by {message.model}</div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={fileUploadMode ? `Ask about "${currentFile?.filename}"...` : "Ask Takku anything... (Press Enter to send)"}
            rows={3}
            disabled={loading}
          />
          <button 
            onClick={sendMessage} 
            disabled={!input.trim() || loading}
            className="send-button"
          >
            {loading ? '⏳' : '📤'}
          </button>
        </div>
        <div className="disclaimer">
          <small>⚠️ AI assistant - Always verify important information from reliable sources.</small>
        </div>
      </div>
    </div>
  );
};

export default TakkuChat;