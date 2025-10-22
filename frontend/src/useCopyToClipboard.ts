import { useState } from 'react';

export const useCopyToClipboard = () => {
  const [isCopied, setIsCopied] = useState(false);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setIsCopied(true);
      
      // Reset after 2 seconds
      setTimeout(() => setIsCopied(false), 2000);
      
      return true;
    } catch (err) {
      console.error('Failed to copy text: ', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
      return true;
    }
  };

  return { isCopied, copyToClipboard };
};