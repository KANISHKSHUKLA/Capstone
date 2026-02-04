'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import React from 'react';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const router = useRouter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    setJobId(data.job_id);
    router.push(`/results/${data.job_id}`);
  };

  return (
    <main className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-primary-600">Upload Research Paper</h1>
        </div>
      </header>

      <div className="flex-grow flex items-center justify-center p-6 sm:p-24">
        <div className="w-full max-w-xl bg-white p-8 rounded-lg shadow-sm border border-neutral-200">
          <form onSubmit={handleSubmit}>
            <div className="space-y-6">
              <div>
                <label 
                  htmlFor="file-upload" 
                  className="block text-sm font-medium text-neutral-700 mb-2"
                >
                  Upload PDF
                </label>
                <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-neutral-300 border-dashed rounded-md">
                  <div className="space-y-1 text-center">
                    <svg
                      className="mx-auto h-12 w-12 text-neutral-400"
                      stroke="currentColor"
                      fill="none"
                      viewBox="0 0 48 48"
                      aria-hidden="true"
                    >
                      <path
                        d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                        strokeWidth={2}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    <div className="flex text-sm text-neutral-600">
                      <label
                        htmlFor="file-upload"
                        className="relative cursor-pointer bg-white rounded-md font-medium text-primary-600 hover:text-primary-500"
                      >
                        <span>Upload a file</span>
                        <input
                          id="file-upload"
                          name="file-upload"
                          type="file"
                          accept=".pdf"
                          className="sr-only"
                          onChange={handleFileChange}
                          required
                        />
                      </label>
                      <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-neutral-500">PDF up to 10MB</p>
                  </div>
                </div>
                {file && (
                  <p className="mt-2 text-sm text-neutral-600">
                    Selected file: {file.name}
                  </p>
                )}
              </div>

              <div className="flex items-center justify-between">
                <Link href="/" className="text-primary-600 hover:text-primary-500 text-sm">
                  Back to Home
                </Link>
                <button
                  type="submit"
                  disabled={!file || isUploading}
                  className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                    !file || isUploading
                      ? 'bg-neutral-300 cursor-not-allowed'
                      : 'bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500'
                  }`}
                >
                  {isUploading ? 'Uploading...' : 'Analyze Paper'}
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </main>
  );
}
