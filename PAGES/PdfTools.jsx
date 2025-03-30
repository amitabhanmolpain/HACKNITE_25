import React, { useState } from 'react';
import { PDFDocument } from 'pdf-lib';

const PdfSummarizer = () => {
  const [summary, setSummary] = useState('');
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSummarize = async () => {
    if (!file) {
      alert('Please select a PDF file.');
      return;
    }

    const arrayBuffer = await file.arrayBuffer();
    const pdfDoc = await PDFDocument.load(arrayBuffer);
    const textContent = await extractTextFromPDF(pdfDoc);
    const summary = summarizeText(textContent);
    setSummary(summary);
  };

  const extractTextFromPDF = async (pdfDoc) => {
    let text = '';
    const pages = pdfDoc.getPages();

    for (const page of pages) {
      const { textContent } = await page.getTextContent();
      text += textContent.items.map(item => item.str).join(' ') + '\n';
    }

    return text;
  };

  const summarizeText = (text) => {
    const sentences = text.split('. ');
    const summaryLength = Math.min(3, sentences.length); // Get the first 3 sentences as a simple summary
    return sentences.slice(0, summaryLength).join('. ') + '.';
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 rounded-lg shadow-md max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-gray-800">PDF Summarizer</h1>
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        className="mb-2 p-2 border border-gray-300 rounded w-full"
      />
      <button 
        onClick={handleSummarize} 
        className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition duration-200"
      >
        Summarize PDF
      </button>
      <div id="summary" className="mt-4 border border-gray-300 p-2 max-h-72 overflow-y-auto w-full bg-white rounded">
        {summary}
      </div>
    </div>
  );
};

export default PdfSummarizer;