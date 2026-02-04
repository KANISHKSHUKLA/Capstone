import React, { useState } from 'react';
import { Card } from './Card';

interface FullExplanationProps {
  data: any;
}

export default function FullExplanation({ data }: FullExplanationProps) {
  const [activeTab, setActiveTab] = useState<string>('approach');
  
  const tabs = [
    { id: 'approach', label: 'Approach' },
    { id: 'methodology', label: 'Methodology' },
    { id: 'architecture', label: 'Architecture' },
    { id: 'results', label: 'Results' },
    { id: 'limitations', label: 'Limitations' }
  ];

  return (
    <Card title="Full Explanation">
      <div>
        <div className="border-b border-neutral-200 mb-4">
          <nav className="-mb-px flex space-x-6 overflow-x-auto pb-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`whitespace-nowrap pb-3 px-1 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-b-2 border-primary-500 text-primary-600'
                    : 'text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="py-2">
          {activeTab === 'approach' && (
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-neutral-800 mb-2">Summary</h4>
                <p className="text-sm text-neutral-700">{data.approach_summary}</p>
              </div>
              
              <div>
                <h4 className="font-medium text-neutral-800 mb-2">Innovations</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
                  {(data?.innovations ?? []).map((innovation: string, index: number) => (
                    <li key={index}>{innovation}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'methodology' && (
            <div>
              <h4 className="font-medium text-neutral-800 mb-2">Methodology</h4>
              <p className="text-sm text-neutral-700">{data.methodology}</p>
            </div>
          )}

          {activeTab === 'architecture' && (
            <div>
              <h4 className="font-medium text-neutral-800 mb-2">Architecture</h4>
              <p className="text-sm text-neutral-700">{data.architecture}</p>
            </div>
          )}

          {activeTab === 'results' && (
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-neutral-800 mb-2">Evaluation</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-3 bg-neutral-50 rounded-md border border-neutral-200">
                    <h5 className="text-xs font-medium text-neutral-500 mb-2">METRICS</h5>
                      <ul className="list-disc pl-4 text-sm text-neutral-700">
                      {(data?.evaluation?.metrics ?? []).map((metric: string, index: number) => (
                        <li key={index}>{metric}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="p-3 bg-neutral-50 rounded-md border border-neutral-200">
                    <h5 className="text-xs font-medium text-neutral-500 mb-2">DATASETS</h5>
                    <ul className="list-disc pl-4 text-sm text-neutral-700">
                      {(data?.evaluation?.datasets ?? []).map((dataset: string, index: number) => (
                        <li key={index}>{dataset}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="p-3 bg-neutral-50 rounded-md border border-neutral-200">
                    <h5 className="text-xs font-medium text-neutral-500 mb-2">BASELINES</h5>
                      <ul className="list-disc pl-4 text-sm text-neutral-700">
                      {(data?.evaluation?.baselines ?? []).map((baseline: string, index: number) => (
                        <li key={index}>{baseline}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-neutral-800 mb-2">Results</h4>
                <p className="text-sm text-neutral-700">{data.results}</p>
              </div>
            </div>
          )}

          {activeTab === 'limitations' && (
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-neutral-800 mb-2">Limitations</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
                  {(Array.isArray(data?.limitations) ? data.limitations : (data?.limitations ? [data.limitations] : [])).map((limitation: string, index: number) => (
                    <li key={index}>{limitation}</li>
                  ))}
                </ul>
              </div>
              
              <div>
                <h4 className="font-medium text-neutral-800 mb-2">Future Work</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
                  {(data?.future_work ?? []).map((work: string, index: number) => (
                    <li key={index}>{work}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
