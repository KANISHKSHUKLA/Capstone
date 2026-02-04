import React, { useState } from 'react';
import { Card } from './Card';

interface ProblemStatementProps {
  data: any;
}

export default function ProblemStatement({ data }: ProblemStatementProps) {
  const [expandedApproach, setExpandedApproach] = useState<number | null>(null);

  const toggleApproach = (index: number) => {
    setExpandedApproach(expandedApproach === index ? null : index);
  };

  const problem = data?.problem ?? 'No problem statement was provided.';
  const researchQuestions: string[] = Array.isArray(data?.research_questions) ? data.research_questions : [];
  const existingApproaches: any[] = Array.isArray(data?.existing_approaches) ? data.existing_approaches : [];
  const gap = data?.gap_in_research ?? 'N/A';
  const importance = data?.importance ?? 'N/A';

  return (
    <Card title="Problem Statement" className="h-full">
      <div className="space-y-4">
        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Research Problem</h4>
          <p className="text-sm text-neutral-700">{problem}</p>
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Research Questions</h4>
          {researchQuestions.length > 0 ? (
            <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
              {researchQuestions.map((question: string, index: number) => (
                <li key={index}>{question}</li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-neutral-600">No research questions listed.</p>
          )}
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Existing Approaches</h4>
          {existingApproaches.length > 0 ? (
            <div className="space-y-2">
              {existingApproaches.map((approach: any, index: number) => (
                <div
                  key={index}
                  className="p-3 bg-neutral-50 rounded-xl border border-neutral-200/70 cursor-pointer hover:bg-white transition-colors"
                  onClick={() => toggleApproach(index)}
                >
                  <div className="flex justify-between items-center gap-3">
                    <h5 className="text-sm font-medium text-neutral-800">
                      {approach?.name ?? 'Untitled approach'}
                    </h5>
                    <span className="text-xs text-neutral-500">
                      {expandedApproach === index ? 'Hide' : 'Show'} limitations
                    </span>
                  </div>

                  {expandedApproach === index && (
                    <div className="mt-2 pt-2 border-t border-neutral-200/70">
                      {(Array.isArray(approach?.limitations) && approach.limitations.length > 0) ? (
                        <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
                          {approach.limitations.map((limitation: string, lIndex: number) => (
                            <li key={lIndex}>{limitation}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-neutral-600">No limitations provided.</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-neutral-600">No existing approaches listed.</p>
          )}
        </div>

        <div className="pt-4 border-t border-neutral-200">
          <h4 className="font-medium text-neutral-800 mb-2">Gap in Research</h4>
          <p className="text-sm text-neutral-700">{gap}</p>
        </div>

        <div className="pt-4 border-t border-neutral-200">
          <h4 className="font-medium text-neutral-800 mb-2">Importance</h4>
          <p className="text-sm text-neutral-700">{importance}</p>
        </div>
      </div>
    </Card>
  );
}
