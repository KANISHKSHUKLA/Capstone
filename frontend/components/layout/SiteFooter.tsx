import React from 'react';

export default function SiteFooter() {
  return (
    <footer className="border-t border-neutral-200/70">
      <div className="mx-auto max-w-7xl px-4 py-8 text-sm text-neutral-500 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <p>
            Built by team 243. 
          </p>
          <p className="text-neutral-400">
            Presented to: <code className="rounded bg-neutral-100 px-1 py-0.5 text-neutral-700">Antima Jain mam</code>.
          </p>
        </div>
      </div>
    </footer>
  );
}

