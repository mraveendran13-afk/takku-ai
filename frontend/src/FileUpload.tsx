import React, { useState, useRef } from 'react';
import axios from 'axios';

interface FileUploadProps {
  onFileProcessed: (filename: string, content: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileProcessed }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    // Clear previous messages
    setError(null);
    setSuccess(null);

    // Validate file type
    const allowedTypes = [
      'application/pdf',
      'text/plain',
      'text/markdown',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];

    if (!allowedTypes.includes(file.type)) {
      setError('Please upload a PDF, TXT, Markdown, or DOCX file.');
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Fixed API URL - uses environment variable or defaults to production
      const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://takku-ai-production.up.railway.app';
      
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true, // Important for CORS with credentials
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            setUploadProgress(progress);
          }
        },
      });

      // Show success message
      setSuccess(response.data.message || `Successfully uploaded ${file.name}!`);
      
      // Call parent callback
      onFileProcessed(response.data.filename, response.data.content_preview);

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
      
    } catch (error: any) {
      console.error('Upload error:', error);
      
      // Better error messages
      if (error.response) {
        setError(error.response.data.detail || 'Error uploading file. Please try again.');
      } else if (error.request) {
        setError('Network error. Please check your connection and try again.');
      } else {
        setError('Error uploading file. Please try again.');
      }
    } finally {
      setUploading(false);
      setUploadProgress(0);
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const triggerFileInput = () => {
    if (!uploading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="file-upload-container">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept=".pdf,.txt,.md,.docx"
        style={{ display: 'none' }}
        disabled={uploading}
      />
      
      <div
        className={`upload-zone ${isDragging ? 'dragging' : ''} ${uploading ? 'uploading' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerFileInput}
        style={{ cursor: uploading ? 'not-allowed' : 'pointer' }}
      >
        {uploading ? (
          <div className="upload-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p>Uploading... {Math.round(uploadProgress)}%</p>
          </div>
        ) : (
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              {isDragging ? 'Drop your file here!' : 'Drag & drop a file or click to browse'}
            </p>
            <p className="upload-subtext">
              Supports PDF, TXT, Markdown, DOCX (max 10MB)
            </p>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="upload-message error">
          <span className="message-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div className="upload-message success">
          <span className="message-icon">✅</span>
          <span>{success}</span>
        </div>
      )}
    </div>
  );
};

export default FileUpload;