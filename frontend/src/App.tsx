import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
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
  sources?: Array<{
    document: string;
    section: number;
    confidence: number;
  }>;
  knowledgeUsed?: boolean;
}

interface CurrentFile {
  filename: string;
  content: string;
}

interface KnowledgeDocument {
  id: string;
  name: string;
  filename: string;
  is_public: boolean;
  uploaded_by: string;
  chunk_count: number;
  upload_date: string;
  tags: string[];
}

// FIXED: Removed /api/v1 from base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://takku-ai-production.up.railway.app';
const ADMIN_PASSWORD = process.env.REACT_APP_ADMIN_PASSWORD || 'takku';

const TakkuChat: React.FC = () => {
  // Use localStorage for persistent chat history
  const [messages, setMessages] = useLocalStorage<Message[]>('takku-chat-history', []);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentFile, setCurrentFile] = useState<CurrentFile | null>(null);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [knowledgeDocuments, setKnowledgeDocuments] = useState<KnowledgeDocument[]>([]);
  const [adminUploading, setAdminUploading] = useState(false);
  
  // FIXED: MANUAL localStorage implementation - guaranteed to work
  const [userId, setUserId] = useState<string>('');

  useEffect(() => {
    // Get or create user ID from localStorage
    let storedUserId = localStorage.getItem('takku-user-id');
    if (!storedUserId) {
      storedUserId = 'user-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('takku-user-id', storedUserId);
      console.log('Created new User ID:', storedUserId);
    }
    setUserId(storedUserId);
    console.log('Using User ID:', storedUserId);
  }, []);
  
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

  const toggleAdminPanel = () => {
  console.log('DEBUG: toggleAdminPanel called');
  console.log('DEBUG: ADMIN_PASSWORD value:', ADMIN_PASSWORD);
  console.log('DEBUG: userId value:', userId);
  console.log('DEBUG: API_BASE_URL:', API_BASE_URL);
  
  setShowAdminPanel(!showAdminPanel);
  if (!showAdminPanel) {
    loadKnowledgeDocuments();
  }
};

  const clearChatHistory = () => {
    if (window.confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
      setMessages([]);
      setCurrentFile(null);
    }
  };

  const loadKnowledgeDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/admin/knowledge-documents`, {
        headers: {
          'X-User-ID': userId,
          'Admin-Password': ADMIN_PASSWORD
        }
      });
      setKnowledgeDocuments(response.data.documents);
    } catch (error) {
      console.error('Failed to load knowledge documents:', error);
      alert('Failed to load knowledge documents. Check admin password.');
    }
  };

  const handleAdminUpload = async (file: File, documentName: string, isPublic: boolean, tags: string) => {
    setAdminUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('document_name', documentName || file.name);
      formData.append('is_public', isPublic.toString());
      formData.append('tags', tags);

      await axios.post(`${API_BASE_URL}/admin/upload-knowledge`, formData, {
        headers: {
          'X-User-ID': userId,
          'Admin-Password': ADMIN_PASSWORD,
          'Content-Type': 'multipart/form-data'
        }
      });

      alert('Document uploaded successfully to knowledge base!');
      await loadKnowledgeDocuments(); // Refresh the list
    } catch (error: any) {
      console.error('Admin upload failed:', error);
      alert(`Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setAdminUploading(false);
    }
  };

  const deleteKnowledgeDocument = async (documentId: string) => {
    if (window.confirm('Are you sure you want to delete this document from the knowledge base?')) {
      try {
        await axios.delete(`${API_BASE_URL}/admin/knowledge-document/${documentId}`, {
          headers: {
            'X-User-ID': userId,
            'Admin-Password': ADMIN_PASSWORD
          }
        });
        alert('Document deleted successfully!');
        await loadKnowledgeDocuments(); // Refresh the list
      } catch (error: any) {
        console.error('Delete failed:', error);
        alert(`Delete failed: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading || !userId) return;

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
      const endpoint = `${API_BASE_URL}/chat`;
      const requestData = {
        message: input,
        symptoms: currentFile ? `File context: ${currentFile.filename}` : "general conversation",
        use_web_search: true
      };

      const response = await axios.post(endpoint, requestData, {
        headers: {
          'X-User-ID': userId
        }
      });

      const aiMessage: Message = { 
        role: 'assistant', 
        content: response.data.response,
        model: response.data.model_used,
        timestamp: Date.now(),
        sources: response.data.sources,
        knowledgeUsed: response.data.knowledge_used
      };
      
      setMessages([...updatedMessages, aiMessage]);
    } catch (error: any) {
      console.error('API Error Details:', error.response?.data || error.message);
      const errorMessage: Message = { 
        role: 'assistant', 
        content: `Sorry, I encountered an error: ${error.response?.data?.detail || error.message}`,
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

  // Format date for display
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
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

  // Admin Upload Component
  const AdminUpload: React.FC = () => {
    const [documentName, setDocumentName] = useState('');
    const [isPublic, setIsPublic] = useState(true);
    const [tags, setTags] = useState('');

    const handleFileUpload = (file: File) => {
      handleAdminUpload(file, documentName, isPublic, tags);
      setDocumentName('');
      setTags('');
    };

    return (
      <div className="admin-upload-panel">
        <h3>📚 Upload to Knowledge Base</h3>
        <div className="upload-form">
          <div className="form-group">
            <label>Document Name:</label>
            <input
              type="text"
              value={documentName}
              onChange={(e) => setDocumentName(e.target.value)}
              placeholder="Enter document name (optional)"
            />
          </div>
          
          <div className="form-group">
            <label>Tags (comma separated):</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="study, biology, notes"
            />
          </div>
          
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
              />
              Make document public (visible to all users)
            </label>
          </div>

          <FileUpload 
            onFileProcessed={(filename, content) => {
              // For admin upload, we use the file directly
              const file = new File([content], filename, { type: 'text/plain' });
              handleFileUpload(file);
            }}
            buttonText={adminUploading ? "Uploading..." : "Upload Document"}
            disabled={adminUploading}
          />

          {adminUploading && (
            <div className="uploading-indicator">
              ⏳ Processing document and generating embeddings...
            </div>
          )}
        </div>

        <div className="documents-list">
          <h4>Knowledge Base Documents</h4>
          {knowledgeDocuments.length === 0 ? (
            <p>No documents in knowledge base yet.</p>
          ) : (
            knowledgeDocuments.map(doc => (
              <div key={doc.id} className="document-item">
                <div className="doc-info">
                  <strong>{doc.name}</strong>
                  <span className={`visibility ${doc.is_public ? 'public' : 'private'}`}>
                    {doc.is_public ? '🌍 Public' : '🔒 Private'}
                  </span>
                  <span className="chunk-count">{doc.chunk_count} chunks</span>
                  <span className="upload-date">{formatDate(doc.upload_date)}</span>
                </div>
                <div className="doc-actions">
                  <button 
                    className="delete-btn"
                    onClick={() => deleteKnowledgeDocument(doc.id)}
                    title="Delete document"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
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
            <div className="header-buttons">
              {messages.length > 0 && (
                <button className="clear-chat-btn" onClick={clearChatHistory}>
                  🗑️ Clear Chat
                </button>
              )}
              <button 
                className={`admin-btn ${showAdminPanel ? 'active' : ''}`}
                onClick={toggleAdminPanel}
              >
                {showAdminPanel ? '📋 Hide Admin' : '⚙️ Knowledge Base'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Admin Panel */}
      {showAdminPanel && <AdminUpload />}

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
            className={`message ${message.role} ${message.isError ? 'error' : ''} ${message.knowledgeUsed ? 'has-knowledge' : ''}`}
          >
            <div className="message-content">
              <div className="message-header">
                <span className="message-role">
                  {message.role === 'user' ? 'You' : 'Takku'}
                  {message.knowledgeUsed && ' 📚'}
                </span>
                <CopyButton content={message.content} />
              </div>
              <div className="message-text">{message.content}</div>
              
              {/* Source Citations */}
              {message.sources && message.sources.length > 0 && (
                <div className="source-citations">
                  <div className="sources-title">📚 Sources:</div>
                  {message.sources.map((source, idx) => (
                    <div key={idx} className="source-item">
                      <span className="source-document">{source.document}</span>
                      <span className="source-section">Section {source.section}</span>
                      <span className="source-confidence">{Math.round(source.confidence * 100)}% match</span>
                    </div>
                  ))}
                </div>
              )}
              
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
              disabled={!input.trim() || loading || !userId}
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