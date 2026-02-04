'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import React from 'react';
import { apiFetch } from '../../lib/api';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await apiFetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Upload failed (${response.status})`);
      }
      const data = await response.json();
      router.push(`/results/${data.job_id}`);
    } catch (err: any) {
      setError(err?.message || 'Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
      <div className="mx-auto max-w-2xl">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-neutral-900">
              Upload a research paper
            </h1>
            <p className="mt-1 text-sm sm:text-base text-neutral-600">
              We’ll extract the text and generate a structured multi-tab analysis.
            </p>
          </div>
          <Link
            href="/"
            className="hidden sm:inline-flex items-center justify-center rounded-2xl border border-neutral-200/80 bg-white/60 px-4 py-2 text-sm font-semibold text-neutral-800 hover:bg-white transition-colors app-ring"
          >
            Back home
          </Link>
        </div>

        <div className="mt-8 app-surface rounded-3xl p-6 sm:p-8 shadow-soft">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="file-upload" className="block text-sm font-semibold text-neutral-800">
                PDF file
              </label>
              <p className="mt-1 text-sm text-neutral-600">
                Choose a PDF (best results with papers under ~10MB).
              </p>

              <div className="mt-4 rounded-3xl border border-dashed border-neutral-300 bg-white/60 p-6">
                <div className="flex flex-col items-center text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary-50 text-primary-700">
                    <svg
                      className="h-6 w-6"
                      stroke="currentColor"
                      fill="none"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path
                        d="M12 16V8m0 0l-3 3m3-3l3 3M20 16.5a4.5 4.5 0 00-3.27-4.32A6 6 0 105 15.75"
                        strokeWidth={2}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>

                  <div className="mt-3 text-sm text-neutral-700">
                    <label
                      htmlFor="file-upload"
                      className="cursor-pointer font-semibold text-primary-700 hover:text-primary-800"
                    >
                      Choose a file
                      <input
                        id="file-upload"
                        name="file-upload"
                        type="file"
                        accept=".pdf"
                        className="sr-only"
                        onChange={handleFileChange}
                        required
                        disabled={isUploading}
                      />
                    </label>{' '}
                    <span className="text-neutral-500">(.pdf)</span>
                  </div>
                  <div className="mt-1 text-xs text-neutral-500">We never modify your file—only analyze it.</div>
                </div>
              </div>

              {file && (
                <div className="mt-4 flex items-center justify-between gap-3 rounded-2xl border border-neutral-200/70 bg-white/60 px-4 py-3">
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-neutral-500">Selected</div>
                    <div className="truncate text-sm font-semibold text-neutral-900">{file.name}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFile(null)}
                    className="shrink-0 rounded-xl border border-neutral-200/80 bg-white px-3 py-2 text-xs font-semibold text-neutral-700 hover:bg-neutral-50 transition-colors app-ring"
                    disabled={isUploading}
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>

            {error && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                {error}
              </div>
            )}

            <div className="flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between">
              <Link href="/" className="text-sm font-semibold text-primary-700 hover:text-primary-800">
                ← Back to home
              </Link>
              <button
                type="submit"
                disabled={!file || isUploading}
                className={`inline-flex items-center justify-center rounded-2xl px-5 py-3 text-sm font-semibold text-white shadow-soft transition-colors app-ring ${
                  !file || isUploading ? 'bg-neutral-300 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700'
                }`}
              >
                {isUploading ? 'Uploading & starting analysis…' : 'Analyze paper'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
