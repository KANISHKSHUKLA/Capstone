'use client';

import React from 'react';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import KeyConcepts from '../../../components/KeyConcepts';
import ProblemStatement from '../../../components/ProblemStatement';
import FullExplanation from '../../../components/FullExplanation';
import PseudoCode from '../../../components/PseudoCode';
import ArchitectureDeepDive from '../../../components/ArchitectureDeepDive';
import ModelFile from '../../../components/ModelFile';
const KnowledgeGraph = dynamic(() => import('../../../components/KnowledgeGraph'), { ssr: false });
import { Loader } from '../../../components/Loader';
import ChatPanel from '../../../components/ChatPanel';
import { apiFetch } from '../../../lib/api';

const LOADING_TIPS = [
  'Getting to know the authors and affiliations…',
  'Scanning the abstract and conclusion…',
  'Mapping out sections and headings…',
  'Extracting key concepts and terminology…',
  'Understanding the core problem and goals…',
  'Reviewing related work and baselines…',
  'Analyzing methodology and experiments…',
  'Tracking datasets, metrics, and benchmarks…',
  'Summarizing results and main findings…',
  'Identifying limitations and future directions…',
  'Preparing the interactive knowledge graph…',
  'Formatting insights for each results tab…',
];

export default function ResultsPage({ params }: { params: { jobId: string } }) {
  const { jobId } = params;
  const [loading, setLoading] = useState<boolean>(true);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('Initializing analysis...');
  const [activeTab, setActiveTab] = useState('summary');
  const [tipIndex, setTipIndex] = useState<number>(0);

  useEffect(() => {
    let pollingInterval: NodeJS.Timeout;
    let progressCounter = 0;
    
    const fetchData = async () => {
      try {
        const response = await apiFetch(`/api/status/${jobId}`);
        
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }
        
        const responseData = await response.json();
        
        if (responseData.status === 'completed') {
          setData(responseData);
          setLoading(false);
          clearInterval(pollingInterval);
        } 
        else if (responseData.status === 'in_progress' || responseData.status === 'processing') {
          setStatusMessage(`Analyzing paper: ${responseData.filename || 'your document'}`);
          
          if (progressCounter < 90) {
            progressCounter += Math.floor(Math.random() * 5) + 1;
            setProgress(Math.min(progressCounter, 90));
          }
        } 
        else if (responseData.status === 'failed') {
          setError(`Analysis failed: ${responseData.error || 'Unknown error'}`);
          setLoading(false);
          clearInterval(pollingInterval);
        }
      } catch (error) {
        console.error('Error fetching paper analysis:', error);
        setError('Failed to connect to analysis server. Please try again later.');
        setLoading(false);
        clearInterval(pollingInterval);
      }
    };

    fetchData();
    pollingInterval = setInterval(fetchData, 3000); 

    const progressInterval = setInterval(() => {
      if (progressCounter < 90) {
        progressCounter += 1;
        setProgress(progressCounter);
      }
    }, 500);

    return () => {
      clearInterval(pollingInterval);
      clearInterval(progressInterval);
    };
  }, [jobId]);

  useEffect(() => {
    if (!loading) return;

    const tipInterval = setInterval(() => {
      setTipIndex((prev) => (prev + 1) % LOADING_TIPS.length);
    }, 2500);

    return () => {
      clearInterval(tipInterval);
    };
  }, [loading]);

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-center justify-between gap-4">
            <Link href="/" className="text-sm font-semibold text-primary-700 hover:text-primary-800">
              ← Back home
            </Link>
            <span className="text-xs text-neutral-500">Job ID: {jobId}</span>
          </div>

          <div className="mt-6 app-surface rounded-3xl p-6 sm:p-10 shadow-soft">
            <Loader message={statusMessage} />

            <div className="mt-8 max-w-md mx-auto">
              <div className="w-full bg-neutral-200/80 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-primary-600 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between mt-2 text-xs text-neutral-500">
                <span>Upload</span>
                <span>Analyze</span>
                <span>Format</span>
              </div>

              <div className="mt-6 text-center text-sm text-neutral-600">
                <div className="font-medium text-neutral-800 mb-1">
                  {LOADING_TIPS[tipIndex]}
                </div>
                <div className="text-xs text-neutral-500">
                  These steps are happening behind the scenes while we analyze your paper.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!data || !data.result) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-14 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-neutral-900">
            {error ? 'Something went wrong' : 'Analysis not found'}
          </h1>
          <p className="mt-2 text-neutral-600">
            {error
              ? error
              : 'The requested analysis could not be found, or it hasn’t completed yet.'}
          </p>
          <div className="mt-6 flex justify-center">
            <Link
              href="/upload"
              className="inline-flex items-center justify-center rounded-2xl bg-primary-600 px-5 py-3 text-sm font-semibold text-white shadow-soft hover:bg-primary-700 transition-colors app-ring"
            >
              Analyze another paper
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const { metadata, key_concepts, problem_statement, full_explanation, pseudo_code, knowledge_graph, architecture_deep_dive, model_file } = data.result;
  
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-10 lg:px-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Paper</div>
          <h1 className="mt-1 text-balance text-2xl sm:text-3xl font-semibold tracking-tight text-neutral-900">
            {metadata?.title || 'Untitled'}
          </h1>
          <p className="mt-2 text-sm text-neutral-600">{metadata?.authors || 'Unknown authors'}</p>
        </div>
        <div className="flex flex-col gap-2 sm:items-end">
          <Link
            href="/upload"
            className="inline-flex items-center justify-center rounded-2xl bg-primary-600 px-4 py-2.5 text-sm font-semibold text-white shadow-soft hover:bg-primary-700 transition-colors app-ring"
          >
            Analyze another
          </Link>
          <div className="text-xs text-neutral-400">Job ID: {jobId}</div>
        </div>
      </div>

      {error && (
        <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="mt-8">
        <div className="flex flex-wrap gap-2">
          {[
            { id: 'summary', label: 'Summary' },
            { id: 'knowledge-graph', label: 'Knowledge graph' },
            { id: 'implementation', label: 'Implementation' },
            { id: 'architecture-deep-dive', label: 'Architecture' },
            { id: 'model-file', label: 'Model file' },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`rounded-2xl px-3.5 py-2 text-sm font-semibold transition-colors app-ring ${
                activeTab === t.id
                  ? 'bg-primary-600 text-white shadow-soft'
                  : 'border border-neutral-200/80 bg-white/60 text-neutral-700 hover:bg-white'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="mt-6 pb-20">
        {activeTab === 'summary' && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <KeyConcepts data={key_concepts} />
              <ProblemStatement data={problem_statement} />
            </div>

            <div className="mt-6">
              <FullExplanation data={full_explanation} />
            </div>
          </>
        )}

        {activeTab === 'knowledge-graph' && (
          <div className="app-card rounded-2xl shadow-soft overflow-hidden">
            <div className="px-6 py-4 border-b border-neutral-200/70 bg-white/50">
              <h2 className="text-base sm:text-lg font-semibold text-neutral-900">Knowledge graph</h2>
              <p className="mt-1 text-sm text-neutral-600">
                Visual map of concepts and relationships. Click nodes to inspect details.
              </p>
            </div>
            <div className="p-4 sm:p-6">
              <KnowledgeGraph graphData={knowledge_graph || { nodes: [], edges: [] }} height={600} />
            </div>
          </div>
        )}

        {activeTab === 'implementation' && (
          <PseudoCode data={pseudo_code} />
        )}

        {activeTab === 'architecture-deep-dive' && (
          <ArchitectureDeepDive data={architecture_deep_dive} />
        )}

        {activeTab === 'model-file' && (
          <ModelFile code={typeof model_file === 'string' ? model_file : (model_file?.code || '')} />
        )}
      </div>
      
      {/* Chat Panel */}
      <ChatPanel jobId={jobId} />
    </div>
  );
}
