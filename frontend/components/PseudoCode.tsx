import React, { useState } from 'react';
import { Card } from './Card';

interface PseudoCodeProps {
  data: any;
}

export default function PseudoCode({ data }: PseudoCodeProps) {
  const [activeComponent, setActiveComponent] = useState<number>(0);

  return (
    <Card title="Implementation Details">
      <div className="space-y-4">
        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Overview</h4>
          <p className="text-sm text-neutral-700">{data.implementation_overview}</p>
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Prerequisites</h4>
          <div className="flex flex-wrap gap-2">
            {data.prerequisites.map((prereq: string, index: number) => (
              <span
                key={index}
                className="px-2 py-1 bg-neutral-100 text-neutral-700 text-xs font-medium rounded-full"
              >
                {prereq}
              </span>
            ))}
          </div>
        </div>

        <div className="mb-4">
          <h4 className="font-medium text-neutral-800 mb-2">Main Components</h4>
          <div className="flex flex-wrap gap-2">
            {data.main_components.map((component: string, index: number) => (
              <span
                key={index}
                className={`px-2 py-1 text-xs font-medium rounded-full cursor-pointer ${
                  activeComponent === index
                    ? 'bg-primary-100 text-primary-700'
                    : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                }`}
                onClick={() => setActiveComponent(index)}
              >
                {component}
              </span>
            ))}
          </div>
        </div>

        <div className="border-t border-neutral-200 pt-4">
          {data.pseudo_code.map((component: any, index: number) => (
            <div key={index} className={activeComponent === index ? 'block' : 'hidden'}>
              <div className="flex justify-between items-start mb-3">
                <h4 className="font-medium text-neutral-800">{component.component}</h4>
                <span className="text-xs px-2 py-0.5 bg-primary-100 text-primary-700 rounded-full">
                  Code
                </span>
              </div>
              <p className="text-sm text-neutral-600 mb-3">{component.description}</p>
              <div className="bg-neutral-800 rounded-md overflow-hidden">
                <pre className="p-4 overflow-x-auto">
                  <code className="text-neutral-100 text-xs">{component.code}</code>
                </pre>
              </div>
            </div>
          ))}
        </div>

        <div className="border-t border-neutral-200 pt-4">
          <h4 className="font-medium text-neutral-800 mb-2">Potential Challenges</h4>
          <ul className="list-disc pl-5 space-y-2 text-sm text-neutral-700">
            {data.potential_challenges.map((item: any, index: number) => (
              <li key={index}>
                {typeof item === 'string' ? (
                  item
                ) : (
                  <>
                    <span className="font-medium">{item.challenge}</span>
                    {item.description && (
                      <span className="text-neutral-600"> - {item.description}</span>
                    )}
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Card>
  );
}
