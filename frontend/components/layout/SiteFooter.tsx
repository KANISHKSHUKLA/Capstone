import React from 'react';

export default function SiteFooter() {
  return (
    <footer className="border-t border-neutral-200/70">
      <div className="mx-auto max-w-7xl px-4 py-8 text-sm text-neutral-500 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <p>
            Built with Next.js + FastAPI.
          </p>
          <p className="text-neutral-400">
            Tip: set <code className="rounded bg-neutral-100 px-1 py-0.5 text-neutral-700">NEXT_PUBLIC_API_BASE_URL</code> for deployments.
          </p>
        </div>
      </div>
    </footer>
  );
}

