import React, { useState } from 'react';
import { Card } from './Card';

interface PseudoCodeProps {
  data: any;
}

export default function PseudoCode({ data }: PseudoCodeProps) {
  const [activeComponent, setActiveComponent] = useState<number>(0);
  const prerequisites: string[] = Array.isArray(data?.prerequisites) ? data.prerequisites : [];
  const mainComponents: string[] = Array.isArray(data?.main_components) ? data.main_components : [];
  const pseudoCodeItems: any[] = Array.isArray(data?.pseudo_code) ? data.pseudo_code : [];
  const potentialChallenges: any[] = Array.isArray(data?.potential_challenges) ? data.potential_challenges : [];

  return (
    <Card title="Implementation Details">
      <div className="space-y-4">
        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Overview</h4>
          <p className="text-sm text-neutral-700">{data?.implementation_overview ?? 'No overview provided.'}</p>
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Prerequisites</h4>
          {prerequisites.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {prerequisites.map((prereq: string, index: number) => (
                <span
                  key={index}
                  className="px-2.5 py-1 bg-neutral-100 text-neutral-700 text-xs font-semibold rounded-full"
                >
                  {prereq}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm text-neutral-600">No prerequisites listed.</p>
          )}
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Main Components</h4>
          {mainComponents.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {mainComponents.map((component: string, index: number) => (
                <button
                  type="button"
                  key={index}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-full transition-colors app-ring ${
                    activeComponent === index
                      ? 'bg-primary-600 text-white shadow-sm'
                      : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                  }`}
                  onClick={() => setActiveComponent(index)}
                >
                  {component}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-sm text-neutral-600">No components listed.</p>
          )}
        </div>

        <div className="border-t border-neutral-200 pt-4">
          {pseudoCodeItems.length > 0 ? (
            pseudoCodeItems.map((component: any, index: number) => (
              <div key={index} className={activeComponent === index ? 'block' : 'hidden'}>
                <div className="flex justify-between items-start mb-3 gap-3">
                  <h4 className="font-medium text-neutral-800">
                    {component?.component ?? `Component ${index + 1}`}
                  </h4>
                  <span className="text-xs px-2 py-0.5 bg-primary-100 text-primary-700 rounded-full font-semibold">
                    Code
                  </span>
                </div>
                <p className="text-sm text-neutral-600 mb-3">{component?.description ?? ''}</p>
                <div className="rounded-xl overflow-hidden border border-neutral-200/70 bg-neutral-900">
                  <pre className="p-4 overflow-x-auto">
                    <code className="text-neutral-100 text-xs leading-relaxed">
                      {component?.code ?? ''}
                    </code>
                  </pre>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-neutral-600">No pseudo-code generated.</p>
          )}
        </div>

        <div className="border-t border-neutral-200 pt-4">
          <h4 className="font-medium text-neutral-800 mb-2">Potential Challenges</h4>
          {potentialChallenges.length > 0 ? (
            <ul className="list-disc pl-5 space-y-2 text-sm text-neutral-700">
              {potentialChallenges.map((item: any, index: number) => (
                <li key={index}>
                  {typeof item === 'string' ? (
                    item
                  ) : (
                    <>
                      <span className="font-medium">{item?.challenge ?? 'Challenge'}</span>
                      {item?.description && (
                        <span className="text-neutral-600"> - {item.description}</span>
                      )}
                    </>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-neutral-600">No challenges listed.</p>
          )}
        </div>
      </div>
    </Card>
  );
}
