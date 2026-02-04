import React, { useState } from 'react';
import { Card } from './Card';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface ArchitectureDeepDiveProps {
  data: any;
}

// Helper function to convert backtick math notation to LaTeX
const convertBackticksToLatex = (text: any): string => {
  // Handle non-string types
  if (!text) return '';
  if (typeof text !== 'string') {
    // If it's an array, join with line breaks
    if (Array.isArray(text)) {
      return text.map(item => 
        typeof item === 'string' ? item : JSON.stringify(item)
      ).join('\n\n');
    }
    // If it's an object, stringify it
    if (typeof text === 'object') {
      return JSON.stringify(text, null, 2);
    }
    // Convert to string for other types
    return String(text);
  }
  
  // Convert backtick-wrapped content that looks like math to LaTeX inline math
  // Pattern: `variable_name` or `equation = something`
  return text.replace(/`([^`]+)`/g, (match, content) => {
    // Check if it looks like math (contains =, +, -, *, /, ^, subscripts, Greek letters, etc.)
    if (/[=+\-*/^_{}∑∏∫σΣΠΘΦλμτ()[\]→←]/.test(content)) {
      // Replace _ with proper LaTeX subscript syntax if not already escaped
      let latexContent = content;
      
      // Handle arrow notation for directional variables (e.g., →h_k,j or ←h_k,j)
      // Convert to proper LaTeX with overrightarrow
      latexContent = latexContent.replace(/→([A-Za-z])/g, '\\overrightarrow{$1}');
      latexContent = latexContent.replace(/←([A-Za-z])/g, '\\overleftarrow{$1}');
      
      // General replacements
      latexContent = latexContent
        .replace(/([A-Za-z]+)_([A-Za-z0-9,{}]+)/g, '$1_{$2}')  // subscripts
        .replace(/\*/g, ' \\cdot ')  // multiplication
        .replace(/⊙/g, ' \\odot ')  // element-wise multiplication
        .replace(/σ/g, '\\sigma')  // Greek letters
        .replace(/Σ/g, '\\Sigma')
        .replace(/→/g, ' \\rightarrow ')  // standalone arrows with spaces
        .replace(/←/g, ' \\leftarrow ');
      
      return `$${latexContent}$`;
    }
    // If it doesn't look like math, keep the code formatting
    return match;
  });
}

export default function ArchitectureDeepDive({ data }: ArchitectureDeepDiveProps) {
  const [activeComponent, setActiveComponent] = useState<number>(0);

  if (!data || !data.detailed_breakdown || data.detailed_breakdown.length === 0) {
    return (
      <Card title="Architecture Deep Dive">
        <p className="text-neutral-600 text-sm">No detailed architecture analysis available.</p>
      </Card>
    );
  }

  return (
    <Card title="🔬 Architecture Deep Dive">
      <div className="space-y-6">
        {/* Overview Section */}
        {data.overview && (
          <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
            <h4 className="font-semibold text-primary-900 mb-2 flex items-center">
              <span className="text-lg mr-2">📋</span>
              Overview
            </h4>
            <p className="text-sm text-primary-800 leading-relaxed">{data.overview}</p>
          </div>
        )}

        {/* Component Navigation */}
        <div>
          <h4 className="font-semibold text-neutral-800 mb-3">Architecture Components</h4>
          <div className="flex flex-wrap gap-2">
            {data.detailed_breakdown.map((component: any, index: number) => (
              <button
                key={index}
                onClick={() => setActiveComponent(index)}
                className={`px-3 py-2 text-sm font-medium rounded-lg transition-all ${
                  activeComponent === index
                    ? 'bg-primary-600 text-white shadow-md'
                    : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                }`}
              >
                {component.component_name}
              </button>
            ))}
          </div>
        </div>

        {/* Active Component Details */}
        <div className="border-t border-neutral-200 pt-6">
          {data.detailed_breakdown.map((component: any, index: number) => (
            <div key={index} className={activeComponent === index ? 'block space-y-4' : 'hidden'}>
              {/* Component Header */}
              <div className="mb-4">
                <h3 className="text-xl font-bold text-neutral-900 mb-2">
                  {component.component_name}
                </h3>
                {component.purpose && (
                  <p className="text-neutral-700 italic">{component.purpose}</p>
                )}
              </div>

              {/* Detailed Explanation */}
              {component.detailed_explanation && (
                <div className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
                  <h5 className="font-semibold text-neutral-800 mb-2 flex items-center">
                    <span className="text-base mr-2">💡</span>
                    Detailed Explanation
                  </h5>
                  <div className="text-sm text-neutral-700 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {convertBackticksToLatex(component.detailed_explanation)}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Mathematical Formulation */}
              {component.mathematical_formulation && (
                <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                  <h5 className="font-semibold text-indigo-900 mb-2 flex items-center">
                    <span className="text-base mr-2">📐</span>
                    Mathematical Formulation
                  </h5>
                  <div className="text-sm text-indigo-900 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {convertBackticksToLatex(component.mathematical_formulation)}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Dimension Analysis */}
              {component.dimension_analysis && (
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <h5 className="font-semibold text-blue-900 mb-2 flex items-center">
                    <span className="text-base mr-2">📊</span>
                    Dimension Analysis
                  </h5>
                  <div className="text-sm text-blue-900 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {convertBackticksToLatex(component.dimension_analysis)}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Design Rationale */}
              {component.design_rationale && (
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <h5 className="font-semibold text-green-900 mb-2 flex items-center">
                    <span className="text-base mr-2">🎯</span>
                    Design Rationale
                  </h5>
                  <div className="text-sm text-green-900 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {component.design_rationale}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Subtle Details */}
              {component.subtle_details && (
                <div className="bg-amber-50 rounded-lg p-4 border border-amber-200">
                  <h5 className="font-semibold text-amber-900 mb-2 flex items-center">
                    <span className="text-base mr-2">⚡</span>
                    Critical Implementation Details
                  </h5>
                  <div className="text-sm text-amber-900 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {component.subtle_details}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Integration Flow */}
        {data.integration_flow && (
          <div className="border-t border-neutral-200 pt-6">
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <h4 className="font-semibold text-purple-900 mb-3 flex items-center">
                <span className="text-lg mr-2">🔄</span>
                End-to-End Integration Flow
              </h4>
              <div className="text-sm text-purple-900 leading-relaxed prose prose-sm max-w-none">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {convertBackticksToLatex(data.integration_flow)}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}

        {/* Critical Insights */}
        {data.critical_insights && data.critical_insights.length > 0 && (
          <div className="border-t border-neutral-200 pt-6">
            <h4 className="font-semibold text-neutral-800 mb-3 flex items-center">
              <span className="text-lg mr-2">💎</span>
              Critical Insights
            </h4>
            <ul className="space-y-2">
              {data.critical_insights.map((insight: string, index: number) => (
                <li key={index} className="flex items-start">
                  <span className="text-primary-600 mr-2 mt-1">▸</span>
                  <span className="text-sm text-neutral-700 leading-relaxed">{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Implementation Considerations */}
        {data.implementation_considerations && data.implementation_considerations.length > 0 && (
          <div className="border-t border-neutral-200 pt-6">
            <h4 className="font-semibold text-neutral-800 mb-3 flex items-center">
              <span className="text-lg mr-2">🛠️</span>
              Implementation Considerations
            </h4>
            <ul className="space-y-2">
              {data.implementation_considerations.map((consideration: string, index: number) => (
                <li key={index} className="flex items-start">
                  <span className="text-amber-600 mr-2 mt-1">⚠️</span>
                  <span className="text-sm text-neutral-700 leading-relaxed">{consideration}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}

