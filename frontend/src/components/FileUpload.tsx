import React, { useState } from 'react';

interface FileUploadProps {
  onFileProcessed: (filename: string, content: string) => void;
  buttonText?: string;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onFileProcessed, 
  buttonText = "Upload File", 
  disabled = false 
}) => {
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);

    try {
      const text = await readFileAsText(file);
      onFileProcessed(file.name, text);
    } catch (error) {
      console.error('File processing error:', error);
      alert('Failed to process file. Please try again.');
    } finally {
      setIsProcessing(false);
      // Reset the input
      event.target.value = '';
    }
  };

  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  return (
    <div className="file-upload">
      <input
        type="file"
        id="file-upload-input"
        accept=".pdf,.txt,.docx,.md"
        onChange={handleFileChange}
        disabled={disabled || isProcessing}
        style={{ display: 'none' }}
      />
      <label 
        htmlFor="file-upload-input" 
        className={`upload-button ${disabled || isProcessing ? 'disabled' : ''}`}
      >
        {isProcessing ? 'Processing...' : buttonText}
      </label>
      {isProcessing && (
        <div className="processing-indicator">
          ⏳ Reading file...
        </div>
      )}
    </div>
  );
};

export default FileUpload;