import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './FileUpload';
import { useSpeechRecognition } from './useSpeechRecognition';
import { useLocalStorage } from './useLocalStorage';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  isError?: boolean;
  timestamp?: number;
}

interface CurrentFile {
  filename: string;
  content: string;
}

interface UserPreferences {
  userName: string;
  preferredLanguage: string;
  theme: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://takku-ai-production.up.railway.app/api/v1';

const TakkuChat: React.FC = () => {
  // Use localStorage for persistent chat history
  const [messages, setMessages] = useLocalStorage<Message[]>('takku-chat-history', []);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentFile, setCurrentFile] = useState<CurrentFile | null>(null);
  const [fileUploadMode, setFileUploadMode] = useState(false);
  const [showFileUpload, setShowFileUpload] = useState(false);
  
  // User preferences stored in localStorage
  const [userPreferences, setUserPreferences] = useLocalStorage<UserPreferences>('takku-user-preferences', {
    userName: '',
    preferredLanguage: 'auto',
    theme: 'default'
  });

  const [suggestions] = useState([
    "What's the best way to learn programming?",
    "Tell me a fun fact about space!",
    "How can I be more productive?",
    "What are some good books to read?",
    "Explain quantum computing in simple terms",
    "What's your favorite superhero movie?"
  ]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Speech recognition hook
  const {
    isListening,
    transcript,
    isSupported,
    startListening,
    stopListening,
  } = useSpeechRecognition();

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update input when speech is recognized
  useEffect(() => {
    if (transcript) {
      setInput(transcript);
    }
  }, [transcript]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleFileProcessed = (filename: string, content: string) => {
    setCurrentFile({ filename, content });
    setFileUploadMode(true);
    setShowFileUpload(false);
    
    const fileMessage: Message = {
      role: 'assistant',
      content: `📁 I've loaded "${filename}"! Now you can ask me questions about this document. What would you like to know? 🐱`,
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, fileMessage]);
  };

  const exitFileMode = () => {
    setCurrentFile(null);
    setFileUploadMode(false);
    
    const exitMessage: Message = {
      role: 'assistant', 
      content: "Switched back to regular chat mode! What would you like to talk about? 🐱",
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, exitMessage]);
  };

  const toggleFileUpload = () => {
    setShowFileUpload(!showFileUpload);
  };

  const clearChatHistory = () => {
    if (window.confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
      setMessages([]);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = { 
      role: 'user', 
      content: input,
      timestamp: Date.now()
    };
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
        model: response.data.model,
        timestamp: Date.now()
      };
      
      setMessages([...updatedMessages, aiMessage]);
    } catch (error) {
      const errorMessage: Message = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.',
        isError: true,
        timestamp: Date.now()
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

  // Format timestamp for display
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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
            {messages.length > 0 && (
              <button className="clear-chat-btn" onClick={clearChatHistory}>
                🗑️ Clear Chat History
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h3>Welcome to Takku AI!</h3>
            <p>Ask me anything! I'm your friendly AI bud here to help with questions, advice, or just chat!</p>

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
              <div className="message-meta">
                {message.model && (
                  <span className="message-model">Generated by {message.model}</span>
                )}
                {message.timestamp && (
                  <span className="message-time">{formatTime(message.timestamp)}</span>
                )}
              </div>
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

      {/* File Upload Area - Shows when attachment button is clicked */}
      {showFileUpload && (
        <div className="file-upload-area">
          <div className="file-upload-header">
            <h4>📁 Upload a document</h4>
            <button className="close-upload" onClick={() => setShowFileUpload(false)}>
              ✕
            </button>
          </div>
          <FileUpload onFileProcessed={handleFileProcessed} />
        </div>
      )}

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
          <div className="input-buttons">
            {isSupported && (
              <button 
                className={`microphone-button ${isListening ? 'listening' : ''}`}
                onClick={isListening ? stopListening : startListening}
                disabled={loading}
                title={isListening ? 'Stop recording' : 'Start voice input'}
              >
                {isListening ? '⏹️' : '🎤'}
              </button>
            )}
            <button 
              className="attachment-button"
              onClick={toggleFileUpload}
              disabled={loading}
              title="Attach file"
            >
              📎
            </button>
            <button 
              onClick={sendMessage} 
              disabled={!input.trim() || loading}
              className="send-button"
            >
              {loading ? '⏳' : '📤'}
            </button>
          </div>
        </div>
        <div className="disclaimer">
          <small>⚠️ AI assistant - Chat history is stored locally in your browser</small>
        </div>
      </div>
    </div>
  );
};

export default TakkuChat;