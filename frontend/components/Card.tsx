import React from 'react';

interface CardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function Card({ title, children, className = '' }: CardProps) {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-neutral-200 overflow-hidden ${className}`}>
      <div className="px-6 py-4 bg-primary-50 border-b border-neutral-200">
        <h3 className="text-lg font-semibold text-primary-700">{title}</h3>
      </div>
      <div className="p-6">{children}</div>
    </div>
  );
}
