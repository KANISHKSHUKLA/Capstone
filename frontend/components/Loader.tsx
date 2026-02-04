import React from 'react';

interface LoaderProps {
  message?: string;
}

export function Loader({ message = 'Processing your paper...' }: LoaderProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative w-24 h-24 mb-8">
        {/* Background circle */}
        <div className="absolute inset-0 rounded-full bg-primary-100"></div>
        
        {/* Spinner circles */}
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary-500 animate-spin"></div>
        <div className="absolute inset-2 rounded-full border-4 border-transparent border-t-primary-400 animate-spin" style={{ animationDuration: '1.5s' }}></div>
        <div className="absolute inset-4 rounded-full border-4 border-transparent border-t-primary-300 animate-spin" style={{ animationDuration: '2s' }}></div>
        
        {/* Center dot */}
        <div className="absolute inset-1/3 rounded-full bg-primary-200"></div>
      </div>
      
      <div className="text-center">
        <h3 className="text-lg font-medium text-neutral-800 mb-2">{message}</h3>
        <p className="text-sm text-neutral-600 max-w-md">
          Our AI is analyzing the research paper. This process usually takes 20-30 seconds.
        </p>
      </div>
    </div>
  );
}
