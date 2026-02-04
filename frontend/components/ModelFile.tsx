import React from 'react';
import { Card } from './Card';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface ModelFileProps {
  code: string;
}

export default function ModelFile({ code }: ModelFileProps) {
  const handleDownload = () => {
    try {
      const blob = new Blob([code || ''], { type: 'text/x-python' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model.py';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      // ignore
    }
  };

  if (!code || code.trim().length === 0) {
    return (
      <Card title="Model File (model.py)">
        <p className="text-sm text-neutral-600">No model.py code was generated.</p>
      </Card>
    );
  }

  return (
    <Card title="Model File (model.py)">
      <div className="mb-4 flex justify-between items-center">
        <p className="text-sm text-neutral-600">
          Complete PyTorch implementation with dimension comments
        </p>
        <button
          onClick={handleDownload}
          className="px-3 py-1.5 bg-primary-600 text-white text-sm rounded hover:bg-primary-700 transition-colors"
        >
          📥 Download model.py
        </button>
      </div>
      <div className="rounded-lg overflow-hidden border border-neutral-200">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          showLineNumbers
          customStyle={{
            margin: 0,
            borderRadius: '0.5rem',
            fontSize: '0.875rem',
            lineHeight: '1.5',
          }}
          codeTagProps={{
            style: {
              fontFamily: 'Menlo, Monaco, Consolas, "Courier New", monospace',
            }
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </Card>
  );
}


