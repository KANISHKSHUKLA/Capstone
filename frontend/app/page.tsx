import React from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
      <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-neutral-200/70 bg-white/60 px-3 py-1 text-xs font-medium text-neutral-600 shadow-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-primary-500" />
            AI paper analysis + chat, in one place
          </div>

          <h1 className="mt-4 text-balance text-4xl font-semibold tracking-tight text-neutral-900 sm:text-5xl">
            Understand research papers <span className="text-primary-700">in minutes</span>, not hours.
          </h1>
          <p className="mt-4 max-w-xl text-pretty text-base leading-relaxed text-neutral-600 sm:text-lg">
            Upload a PDF and get a structured breakdown: key concepts, problem statement, methodology, implementation pseudo-code,
            an architecture deep dive, a generated <code className="rounded bg-white/70 px-1 py-0.5">model.py</code>, and a chat panel to ask follow-ups.
          </p>

          <div className="mt-7 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link
              href="/upload"
              className="inline-flex items-center justify-center rounded-2xl bg-primary-600 px-5 py-3 text-sm font-semibold text-white shadow-soft hover:bg-primary-700 transition-colors app-ring"
            >
              Upload a paper
            </Link>
            <Link
              href="/upload"
              className="inline-flex items-center justify-center rounded-2xl border border-neutral-200/80 bg-white/60 px-5 py-3 text-sm font-semibold text-neutral-800 hover:bg-white transition-colors app-ring"
            >
              Go to analyzer
            </Link>
          </div>

          <div className="mt-10 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {[
              { title: 'Structured summary', desc: 'Concepts, problem, methodology, results—easy to scan.' },
              { title: 'Implementation view', desc: 'Pseudo-code + challenges for building it.' },
              { title: 'Architecture deep dive', desc: 'Math/shape details with nicely rendered markdown.' },
              { title: 'Paper chat', desc: 'Ask questions without leaving the results.' },
            ].map((f) => (
              <div key={f.title} className="app-surface rounded-2xl px-4 py-4">
                <div className="text-sm font-semibold text-neutral-900">{f.title}</div>
                <div className="mt-1 text-sm text-neutral-600">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="app-surface rounded-3xl p-6 shadow-soft">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold text-neutral-900">How it works</div>
              <div className="mt-1 text-sm text-neutral-600">A clean, repeatable workflow.</div>
            </div>
            <div className="rounded-2xl bg-primary-50 px-3 py-1 text-xs font-semibold text-primary-700">
              3 steps
            </div>
          </div>

          <ol className="mt-6 space-y-3">
            {[
              { step: 'Upload', text: 'Drop a PDF (research paper).' },
              { step: 'Analyze', text: 'Backend extracts text + runs parallel LLM calls.' },
              { step: 'Explore', text: 'Tabs + chat panel help you dig deeper.' },
            ].map((s, idx) => (
              <li key={s.step} className="flex gap-3 rounded-2xl border border-neutral-200/70 bg-white/60 p-4">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary-600 text-sm font-semibold text-white shadow-sm">
                  {idx + 1}
                </div>
                <div>
                  <div className="text-sm font-semibold text-neutral-900">{s.step}</div>
                  <div className="mt-0.5 text-sm text-neutral-600">{s.text}</div>
                </div>
              </li>
            ))}
          </ol>

          <div className="mt-6 rounded-2xl border border-neutral-200/70 bg-neutral-50 p-4">
            <div className="text-xs font-semibold text-neutral-500">Pro tip</div>
            <div className="mt-1 text-sm text-neutral-700">
              Keep the chat panel open while reading results—it's resizable and stays out of your way.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
