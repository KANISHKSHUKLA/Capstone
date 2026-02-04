import React from 'react';

interface CardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function Card({ title, children, className = '' }: CardProps) {
  return (
    <section className={`app-card rounded-2xl shadow-soft overflow-hidden ${className}`}>
      <div className="px-6 py-4 border-b border-neutral-200/70 bg-white/50">
        <h3 className="text-base sm:text-lg font-semibold text-neutral-900">
          {title}
        </h3>
      </div>
      <div className="p-6">{children}</div>
    </section>
  );
}
