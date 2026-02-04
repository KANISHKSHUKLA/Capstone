import Link from 'next/link';
import React from 'react';

export default function SiteHeader() {
  return (
    <header className="sticky top-0 z-30 border-b border-neutral-200/70 bg-white/70 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <Link href="/" className="group inline-flex items-center gap-2">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-primary-600 text-white shadow-sm">
            <span className="text-sm font-semibold">RP</span>
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-neutral-900 group-hover:text-primary-700 transition-colors">
              Research Paper Summarizer
            </div>
            <div className="text-xs text-neutral-500">Analyze PDFs • Graph • Chat</div>
          </div>
        </Link>

        <nav className="flex items-center gap-2">
          <Link
            href="/upload"
            className="inline-flex items-center justify-center rounded-xl bg-primary-600 px-3.5 py-2 text-sm font-semibold text-white shadow-soft hover:bg-primary-700 transition-colors app-ring"
          >
            Upload
          </Link>
        </nav>
      </div>
    </header>
  );
}

