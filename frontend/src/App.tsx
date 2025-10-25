import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './FileUpload';
import { useSpeechRecognition } from './useSpeechRecognition';
import { useLocalStorage } from './useLocalStorage';
import { useCopyToClipboard } from './useCopyToClipboard';
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

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://takku-ai-production.up.railway.app';

const TakkuChat: React.FC = () => {
  // Use localStorage for persistent chat history
  const [messages, setMessages] = useLocalStorage<Message[]>('takku-chat-history', []);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentFile, setCurrentFile] = useState<CurrentFile | null>(null);
  const [showFileUpload, setShowFileUpload] = useState(false);
  
  // FIXED: Persistent user ID for memory system
  const [userId] = useLocalStorage<string>('takku-user-id', 'user-' + Math.random().toString(36).substr(2, 9));
  
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
    setShowFileUpload(false);
    
    const fileMessage: Message = {
      role: 'assistant',
      content: `📁 I've loaded "${filename}"! You can now ask me questions about this document anytime. 🐱`,
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, fileMessage]);
  };

  const removeCurrentFile = () => {
    setCurrentFile(null);
    
    const removeMessage: Message = {
      role: 'assistant', 
      content: "I've cleared the loaded document. What would you like to talk about now? 🐱",
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, removeMessage]);
  };

  const toggleFileUpload = () => {
    setShowFileUpload(!showFileUpload);
  };

  const clearChatHistory = () => {
    if (window.confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
      setMessages([]);
      setCurrentFile(null);
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
      // FIXED: Use persistent user ID for memory
      const endpoint = `${API_BASE_URL}/chat`;
      const requestData = {
        message: input,
        symptoms: currentFile ? `File context: ${currentFile.filename}` : "general conversation",
        use_web_search: true
      };

      const response = await axios.post(endpoint, requestData, {
        headers: {
          'X-User-ID': userId  // ✅ SAME ID FOR ALL MESSAGES
        }
      });

      const aiMessage: Message = { 
        role: 'assistant', 
        content: response.data.response,
        model: response.data.model_used,
        timestamp: Date.now()
      };
      
      setMessages([...updatedMessages, aiMessage]);
    } catch (error) {
      console.error('API Error:', error);
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

  // Copy to clipboard component for individual messages
  const CopyButton: React.FC<{ content: string }> = ({ content }) => {
    const { isCopied, copyToClipboard } = useCopyToClipboard();

    const handleCopy = () => {
      copyToClipboard(content);
    };

    return (
      <button 
        className={`copy-button ${isCopied ? 'copied' : ''}`}
        onClick={handleCopy}
        title={isCopied ? 'Copied!' : 'Copy to clipboard'}
      >
        {isCopied ? '✅' : '📋'}
      </button>
    );
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
                🗑️ Clear Chat
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Active File Indicator - Always visible when file is loaded */}
      {currentFile && (
        <div className="active-file-indicator">
          <div className="file-info">
            <span className="file-icon">📁</span>
            <span className="file-name">{currentFile.filename}</span>
            <button className="remove-file" onClick={removeCurrentFile} title="Remove document">
              ✕
            </button>
          </div>
          <div className="file-preview">
            <strong>Loaded:</strong> {currentFile.content.substring(0, 100)}...
          </div>
        </div>
      )}

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

        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role} ${message.isError ? 'error' : ''}`}
          >
            <div className="message-content">
              <div className="message-header">
                <span className="message-role">
                  {message.role === 'user' ? 'You' : 'Takku'}
                </span>
                <CopyButton content={message.content} />
              </div>
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
            placeholder={currentFile ? `Ask about "${currentFile.filename}" or anything else...` : "Ask Takku anything... (Press Enter to send)"}
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