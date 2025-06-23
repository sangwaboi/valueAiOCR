import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setDownloadUrl(null);
    setError(null);
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    setUploading(true);
    setDownloadUrl(null);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch('http://localhost:8000/process_pdf/', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to process PDF');
      }
      // Create a blob URL for the returned PDF
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="App">
      <h1>PDF Reconstructor</h1>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
          disabled={uploading}
        />
        <button type="submit" disabled={!selectedFile || uploading}>
          {uploading ? 'Uploading...' : 'Upload & Fix PDF'}
        </button>
      </form>
      {uploading && <div>Processing PDF, please wait...</div>}
      {downloadUrl && (
        <div style={{ marginTop: 20 }}>
          <a href={downloadUrl} download="ordered.pdf">
            <button>Download Ordered PDF</button>
          </a>
        </div>
      )}
      {error && <div style={{ color: 'red' }}>{error}</div>}
    </div>
  );
}

export default App;
