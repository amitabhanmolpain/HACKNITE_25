import React, { useState } from 'react';
import { Upload, FileText } from 'lucide-react';

export default function PdfTools() {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    // Handle file drop here
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 bg-gradient-dark border-b border-gray-800/50">
        <h2 className="text-xl font-semibold text-gray-100">PDF Tools</h2>
      </div>

      <div className="flex-1 p-6">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`h-64 border-2 border-dashed rounded-lg flex flex-col items-center justify-center transition-colors backdrop-blur-sm ${
            isDragging ? 'border-blue-500/50 bg-gray-800/30' : 'border-gray-700/50'
          }`}
        >
          <Upload size={48} className="text-gray-500 mb-4" />
          <p className="text-lg text-gray-300 mb-2">Drag and drop your PDF here</p>
          <p className="text-sm text-gray-500">or</p>
          <label className="mt-4">
            <input type="file" accept=".pdf" className="hidden" />
            <span className="px-4 py-2 bg-blue-600/90 text-white rounded-lg cursor-pointer hover:bg-blue-700/90 transition-colors">
              Browse Files
            </span>
          </label>
        </div>

        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-100">Recent PDFs</h3>
          <div className="space-y-2">
            <div className="p-4 bg-gradient-card border border-gray-800/50 rounded-lg flex items-center gap-3 backdrop-blur-sm">
              <FileText size={24} className="text-blue-400" />
              <div>
                <p className="font-medium text-gray-200">example.pdf</p>
                <p className="text-sm text-gray-500">Uploaded 2 hours ago</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}