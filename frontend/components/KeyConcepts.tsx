import React, { useState } from 'react';
import { Card } from './Card';

interface KeyConceptsProps {
  data: any;
}

export default function KeyConcepts({ data }: KeyConceptsProps) {
  const [expandedConcept, setExpandedConcept] = useState<number | null>(null);

  const toggleExpand = (index: number) => {
    setExpandedConcept(expandedConcept === index ? null : index);
  };

  return (
    <Card title="Key Concepts" className="h-full">
      <div className="space-y-4">
        {/* Provide safe defaults when `data` is undefined or missing fields */}
        <div className="flex flex-wrap gap-2 mb-4">
          {(data?.core_technologies ?? []).map((tech: string, index: number) => (
            <span
              key={index}
              className="px-2 py-1 bg-primary-100 text-primary-700 text-xs font-medium rounded-full"
            >
              {tech}
            </span>
          ))}
        </div>

        <div className="space-y-3">
          {(data?.key_concepts ?? []).map((concept: any, index: number) => (
            <div 
              key={index} 
              className="p-3 bg-neutral-50 rounded-md border border-neutral-200 cursor-pointer hover:bg-neutral-100 transition-colors"
              onClick={() => toggleExpand(index)}
            >
              <div className="flex justify-between items-start">
                <h4 className="font-medium text-neutral-800">{concept?.name ?? 'Untitled'}</h4>
                <span className="text-xs px-2 py-0.5 bg-neutral-200 text-neutral-700 rounded-full">
                  {concept?.category ?? 'general'}
                </span>
              </div>
              
              {expandedConcept === index && (
                <div className="mt-2 pt-2 border-t border-neutral-200">
                  <p className="text-sm text-neutral-700 mb-2">{concept?.explanation ?? 'No explanation provided.'}</p>
                  <p className="text-xs text-neutral-600 italic">
                    <span className="font-medium">Relevance:</span> {concept?.relevance ?? 'N/A'}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="pt-4 border-t border-neutral-200">
          <h4 className="font-medium text-neutral-800 mb-2">Novel Aspects</h4>
          <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
            {(data?.novelty_aspects ?? []).map((aspect: string, index: number) => (
              <li key={index}>{aspect}</li>
            ))}
          </ul>
        </div>
      </div>
    </Card>
  );
}
