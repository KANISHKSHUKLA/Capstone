import React from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-primary-600">Research Paper Summarizer</h1>
        </div>
      </header>

      <div className="flex-grow flex flex-col items-center justify-center p-6 sm:p-24">
        <div className="w-full max-w-xl text-center">
          <h2 className="text-2xl font-semibold mb-6">Analyze Research Papers with AI</h2>
          <p className="mb-8 text-neutral-600">
            Upload a PDF of a research paper and get an in-depth analysis powered by large language models.
          </p>
          <Link href="/upload" 
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
            Upload a Paper
          </Link>
        </div>
      </div>

      <footer className="bg-white border-t border-neutral-200 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-neutral-500 text-sm">
            Powered by FastAPI and DeepSeek LLM
          </p>
        </div>
      </footer>
    </main>
  );
}
