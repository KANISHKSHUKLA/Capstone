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

  return (
    <Card title="Problem Statement" className="h-full">
      <div className="space-y-4">
        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Research Problem</h4>
          <p className="text-sm text-neutral-700">{data.problem}</p>
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Research Questions</h4>
          <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
            {data.research_questions.map((question: string, index: number) => (
              <li key={index}>{question}</li>
            ))}
          </ul>
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Existing Approaches</h4>
          <div className="space-y-2">
            {data.existing_approaches.map((approach: any, index: number) => (
              <div
                key={index}
                className="p-3 bg-neutral-50 rounded-md border border-neutral-200 cursor-pointer hover:bg-neutral-100 transition-colors"
                onClick={() => toggleApproach(index)}
              >
                <div className="flex justify-between items-center">
                  <h5 className="text-sm font-medium text-neutral-800">{approach.name}</h5>
                  <span className="text-xs text-neutral-500">
                    {expandedApproach === index ? 'Hide' : 'Show'} limitations
                  </span>
                </div>

                {expandedApproach === index && (
                  <div className="mt-2 pt-2 border-t border-neutral-200">
                    <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
                      {approach.limitations.map((limitation: string, lIndex: number) => (
                        <li key={lIndex}>{limitation}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="pt-4 border-t border-neutral-200">
          <h4 className="font-medium text-neutral-800 mb-2">Gap in Research</h4>
          <p className="text-sm text-neutral-700">{data.gap_in_research}</p>
        </div>

        <div className="pt-4 border-t border-neutral-200">
          <h4 className="font-medium text-neutral-800 mb-2">Importance</h4>
          <p className="text-sm text-neutral-700">{data.importance}</p>
        </div>
      </div>
    </Card>
  );
}
